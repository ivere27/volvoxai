#ifndef VOLVOXAI_WASM_FREESTANDING_MATH_H
#define VOLVOXAI_WASM_FREESTANDING_MATH_H

/* Clang lowers these builtins to scalar WASM instructions. */
#define isfinite(value) __builtin_isfinite(value)
#define signbit(value) __builtin_signbit(value)
#define copysign(value, sign) __builtin_copysign(value, sign)
#define isnan(value) __builtin_isnan(value)
#define fabs(value) __builtin_fabs(value)
#define trunc(value) __builtin_trunc(value)
#define truncf(value) __builtin_truncf(value)
#define fabsf(value) __builtin_fabsf(value)
#define floorf(value) __builtin_floorf(value)
#define sqrtf(value) __builtin_sqrtf(value)
#define floor(value) __builtin_floor(value)
#define ceil(value) __builtin_ceil(value)
#define sqrt(value) __builtin_sqrt(value)
#define lrintf(value) __builtin_lrintf(value)
#define lrint(value) __builtin_lrint(value)
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))

/* Defined in wasm_math.c. These functions never cross the host transport. */
float cosf(float value);
float expf(float value);
float logf(float value);
float powf(float base, float exponent);
float sinf(float value);
float tanhf(float value);
float erff(float value);
double exp(double value);
double log(double value);
double cos(double value);
double sin(double value);
float expm1f(float value);
double scalbn(double value, int exponent);
double frexp(double value, int* exponent);
double fmod(double value, double divisor);

#endif
