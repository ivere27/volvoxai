/* Scalar math for the freestanding WASM runtime.
 *
 * Range reduction and convergent power series are expressed directly here.
 * Constants are derived by tools/generate_wasm_math.py, independently of any
 * libm source or coefficient collection. Native releases use platform libm;
 * native tests compile this file under renamed symbols for comparison.
 */
#include <stdint.h>
#include <math.h>
#include "wasm_math_constants.h"

typedef union { double value; uint64_t bits; } VxMathDouble;

static double vx_math_scale(double value, int exponent) {
    VxMathDouble input = { .value = value };
    int original = (int)(input.bits >> 52 & 2047);
    int adjustment = 0;
    if (original == 2047 || (input.bits << 1) == 0) return value;
    if (!original) {
        input.value *= 0x1p54;
        original = (int)(input.bits >> 52 & 2047);
        adjustment = -54;
    }
    int64_t target = (int64_t)original + exponent + adjustment;
    uint64_t sign = input.bits & UINT64_C(0x8000000000000000);
    uint64_t fraction = input.bits & UINT64_C(0x000fffffffffffff);
    if (target >= 2047) input.bits = sign | UINT64_C(0x7ff0000000000000);
    else if (target > 0) input.bits = sign | (uint64_t)target << 52 | fraction;
    else if (target < -53) input.bits = sign;
    else {
        /* One floating multiplication rounds directly into the subnormal
         * range. There is no intermediate subnormal rounding. */
        VxMathDouble factor = { .bits = (uint64_t)(target + 1022) << 52 };
        input.bits = sign | UINT64_C(0x0010000000000000) | fraction;
        input.value *= factor.value;
    }
    return input.value;
}

double scalbn(double value, int exponent) { return vx_math_scale(value, exponent); }

static double vx_math_fraction(double value, int* exponent) {
    VxMathDouble input = { .value = value };
    int original = (int)(input.bits >> 52 & 2047), adjustment = 0;
    *exponent = 0;
    if (original == 2047 || (input.bits << 1) == 0) return value;
    if (!original) {
        input.value *= 0x1p54;
        original = (int)(input.bits >> 52 & 2047);
        adjustment = -54;
    }
    *exponent = original - 1022 + adjustment;
    input.bits = (input.bits & UINT64_C(0x800fffffffffffff)) | UINT64_C(0x3fe0000000000000);
    return input.value;
}

double frexp(double value, int* exponent) { return vx_math_fraction(value, exponent); }

double fmod(double value, double divisor) {
    if (!isfinite(value) || isnan(divisor) || divisor == 0.0) return NAN;
    double a = __builtin_fabs(value), b = __builtin_fabs(divisor);
    if (a < b) return value;
    if (a == b) return __builtin_copysign(0.0, value);
    int a_exponent, b_exponent;
    VxMathDouble a_fraction = { .value = vx_math_fraction(a, &a_exponent) };
    VxMathDouble b_fraction = { .value = vx_math_fraction(b, &b_exponent) };
    uint64_t a_integer = (a_fraction.bits & UINT64_C(0x000fffffffffffff)) | UINT64_C(1) << 52;
    uint64_t b_integer = (b_fraction.bits & UINT64_C(0x000fffffffffffff)) | UINT64_C(1) << 52;
    for (;;) {
        if (a_integer >= b_integer) a_integer -= b_integer;
        if (!a_integer || a_exponent == b_exponent) break;
        a_integer <<= 1;
        a_exponent--;
    }
    return __builtin_copysign(vx_math_scale((double)a_integer, a_exponent - 53), value);
}

static double vx_math_polynomial(double value, const double* coefficients, int count) {
    double result = coefficients[count - 1];
    for (int index = count - 2; index >= 0; index--)
        result = coefficients[index] + value * result;
    return result;
}

static double vx_math_exp(double value) {
    if (isnan(value)) return value;
    if (value > 710.0) return INFINITY;
    if (value < -746.0) return 0.0;
    double scaled = value * VX_MATH_INV_LOG_TWO;
    int power = (int)(scaled + (scaled < 0.0 ? -0.5 : 0.5));
    double reduced = (value - power * VX_MATH_LOG_TWO_HIGH) - power * VX_MATH_LOG_TWO_LOW;
    double tail = reduced * reduced * vx_math_polynomial(reduced, vx_math_exp_series, 19);
    return vx_math_scale(1.0 + (reduced + tail), power);
}

double exp(double value) { return vx_math_exp(value); }
float expf(float value) { return (float)vx_math_exp(value); }

static double vx_math_expm1(double value) {
    if (value == 0.0) return value;
    if (__builtin_fabs(value) > 0.5) return vx_math_exp(value) - 1.0;
    return value + value * value * vx_math_polynomial(value, vx_math_exp_series, 19);
}

float expm1f(float value) { return (float)vx_math_expm1(value); }

static double vx_math_log(double value) {
    if (value == 0.0) return -INFINITY;
    if (value < 0.0) return NAN;
    if (!isfinite(value)) return value;
    int power;
    double mantissa = vx_math_fraction(value, &power);
    if (mantissa < VX_MATH_SQRT_HALF) { mantissa *= 2.0; power--; }
    double ratio = (mantissa - 1.0) / (mantissa + 1.0);
    double square = ratio * ratio;
    double logarithm = 2.0 * (ratio + ratio * square *
        vx_math_polynomial(square, vx_math_log_series, 19));
    return power * VX_MATH_LOG_TWO_HIGH + (logarithm + power * VX_MATH_LOG_TWO_LOW);
}

double log(double value) { return vx_math_log(value); }
float logf(float value) { return (float)vx_math_log(value); }

float powf(float base, float exponent) {
    double magnitude = __builtin_fabs((double)base);
    if (exponent == 0.0f || base == 1.0f) return 1.0f;
    if (isnan(base) || isnan(exponent)) return base + exponent;
    if (!isfinite(exponent)) {
        if (magnitude == 1.0) return 1.0f;
        return (magnitude > 1.0) == (exponent > 0.0f) ? INFINITY : 0.0f;
    }
    int integer = __builtin_truncf(exponent) == exponent;
    int odd = integer && __builtin_fabsf(exponent) < 0x1p24f && ((int)exponent & 1);
    int negative = signbit(base) && odd;
    if (magnitude == 0.0 || !isfinite(magnitude)) {
        float result = (magnitude == 0.0) == (exponent > 0.0f) ? 0.0f : INFINITY;
        return negative ? -result : result;
    }
    if (base < 0.0f && !integer) return NAN;
    double result = vx_math_exp((double)exponent * vx_math_log(magnitude));
    return (float)(negative ? -result : result);
}

float tanhf(float value) {
    if (value == 0.0f || isnan(value)) return value;
    double magnitude = __builtin_fabs((double)value);
    double difference = vx_math_expm1(-2.0 * magnitude);
    double result = -difference / (2.0 + difference);
    return (float)__builtin_copysign(result, value);
}

float erff(float value) {
    if (value == 0.0f || isnan(value)) return value;
    double magnitude = __builtin_fabs((double)value);
    if (magnitude >= 4.0) return __builtin_copysignf(1.0f, value);
    /* Positive-term expansion of the defining Gaussian integral after
     * factoring out exp(-x*x); avoids cancellation near erf(x) == 1. */
    double square = magnitude * magnitude, term = magnitude, sum = magnitude;
    for (int index = 1; index < 128; index++) {
        term *= square / (index + 0.5);
        sum += term;
        if (term <= sum * 0x1p-56) break;
    }
    double result = VX_MATH_TWO_OVER_SQRT_PI * vx_math_exp(-square) * sum;
    return (float)__builtin_copysign(result, value);
}

static unsigned vx_math_bit(const uint32_t* words, int index) {
    return index < 0 || index >= 1344 ? 0u : words[index / 32] >> (index % 32) & 1u;
}

static uint64_t vx_math_extract(const uint32_t* words, int top, int count) {
    uint64_t result = 0;
    for (int index = 0; index < count; index++)
        result = result << 1 | vx_math_bit(words, top - index);
    return result;
}

static double vx_math_reduce(double magnitude, unsigned* quadrant) {
    if (magnitude <= VX_MATH_PI_OVER_TWO * 0.5) { *quadrant = 0; return magnitude; }
    if (magnitude < 0x1p20) {
        int multiple = (int)(magnitude * VX_MATH_TWO_OVER_PI + 0.5);
        *quadrant = (unsigned)multiple & 3;
        double reduced = ((magnitude - multiple * VX_MATH_PI_HIGH) -
            multiple * VX_MATH_PI_MIDDLE) - multiple * VX_MATH_PI_LOW;
        /* Near a multiple of pi/2, the last split product's rounding can
         * dominate a tiny remainder. Use the integer reducer there too. */
        if (__builtin_fabs(reduced) >= 0x1p-20) return reduced;
    }
    VxMathDouble input = { .value = magnitude };
    uint64_t mantissa = (input.bits & UINT64_C(0x000fffffffffffff)) | UINT64_C(1) << 52;
    int point = VX_MATH_RECIPROCAL_BITS - ((int)(input.bits >> 52) - 1075);
    uint32_t product[42] = {0};
    /* Multiply the exact 53-bit significand by floor(2/pi * 2^1280).
     * Even DBL_MAX retains over 300 fractional bits before reduction. */
    for (int part = 0; part < 2; part++) {
        uint64_t carry = 0;
        uint32_t multiplier = (uint32_t)(mantissa >> (32 * part));
        for (int index = 0; index < 40; index++) {
            uint64_t sum = (uint64_t)vx_math_reciprocal[index] * multiplier +
                product[index + part] + carry;
            product[index + part] = (uint32_t)sum;
            carry = sum >> 32;
        }
        product[40 + part] = (uint32_t)carry;
    }
    unsigned up = vx_math_bit(product, point - 1);
    *quadrant = (vx_math_bit(product, point) +
        2 * vx_math_bit(product, point + 1) + up) & 3;
    int used = (point + 31) / 32;
    unsigned remainder = (unsigned)point % 32;
    if (remainder) product[used - 1] &= UINT32_MAX >> (32 - remainder);
    if (up) {
        uint64_t carry = 1;
        for (int index = 0; index < used; index++) {
            carry += (uint64_t)(uint32_t)~product[index];
            product[index] = (uint32_t)carry;
            carry >>= 32;
        }
        if (remainder) product[used - 1] &= UINT32_MAX >> (32 - remainder);
    }
    for (int index = used; index < 42; index++) product[index] = 0;
    while (used && !product[used - 1]) used--;
    if (!used) return up ? -0.0 : 0.0;
    int top = (used - 1) * 32 + 31 - __builtin_clz(product[used - 1]);
    double high = vx_math_scale((double)vx_math_extract(product, top, 53), top - 52 - point);
    double low = vx_math_scale((double)vx_math_extract(product, top - 53, 53), top - 105 - point);
    double reduced = high * VX_MATH_PI_OVER_TWO +
        (low * VX_MATH_PI_OVER_TWO + high * VX_MATH_PI_OVER_TWO_TAIL);
    return up ? -reduced : reduced;
}

static double vx_math_trigonometric(double value, int cosine) {
    if (!isfinite(value)) return NAN;
    unsigned quadrant;
    double reduced = vx_math_reduce(__builtin_fabs(value), &quadrant);
    double square = reduced * reduced;
    if (cosine) quadrant = (quadrant + 1) & 3;
    double result = quadrant & 1 ?
        1.0 + square * vx_math_polynomial(square, vx_math_cos_series, 10) :
        reduced + reduced * square * vx_math_polynomial(square, vx_math_sin_series, 9);
    if (quadrant & 2) result = -result;
    if (!cosine && signbit(value)) result = -result;
    return result;
}

double sin(double value) { return vx_math_trigonometric(value, 0); }
double cos(double value) { return vx_math_trigonometric(value, 1); }
float sinf(float value) { return (float)vx_math_trigonometric(value, 0); }
float cosf(float value) { return (float)vx_math_trigonometric(value, 1); }
