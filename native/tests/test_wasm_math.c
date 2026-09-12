/* Independent platform-libm oracle for the C implementation used by WASM.
 * Compare rounded long-double results, including exponent extremes, reduction
 * boundaries, signed zero and the domain rules used by neural operators. */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define scalbn test_scalbn
#define frexp test_frexp
#define fmod test_fmod
#define exp test_exp
#define expf test_expf
#define expm1f test_expm1f
#define log test_log
#define logf test_logf
#define powf test_powf
#define tanhf test_tanhf
#define erff test_erff
#define sin test_sin
#define cos test_cos
#define sinf test_sinf
#define cosf test_cosf
#include "../src/runtime/wasm_math.c"
#undef scalbn
#undef frexp
#undef fmod
#undef exp
#undef expf
#undef expm1f
#undef log
#undef logf
#undef powf
#undef tanhf
#undef erff
#undef sin
#undef cos
#undef sinf
#undef cosf

typedef union { float value; uint32_t bits; } TestFloat;
typedef union { double value; uint64_t bits; } TestDouble;
static uint64_t random_state = UINT64_C(0x132abce546789df0);
static unsigned failures, checks;
static uint64_t worst_float, worst_double;

static uint64_t random_bits(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 7;
    random_state ^= random_state << 17;
    return random_state;
}

static void compare(const char* name, double input, double second,
                    double actual, double expected, int single, int exact) {
    checks++;
    if (isnan(expected) && isnan(actual)) return;
    uint64_t a, b;
    if (single) {
        a = (TestFloat){.value = (float)actual}.bits;
        b = (TestFloat){.value = (float)expected}.bits;
    } else {
        a = (TestDouble){.value = actual}.bits;
        b = (TestDouble){.value = expected}.bits;
    }
    uint64_t distance = a > b ? a - b : b - a;
    uint64_t limit = exact ? 0 : single ? 1 : 4;
    if (!isfinite(expected) || !isfinite(actual) || expected == 0.0 || actual == 0.0)
        limit = 0;
    uint64_t* worst = single ? &worst_float : &worst_double;
    if (!exact && distance > *worst) *worst = distance;
    if (distance > limit && failures++ < 8)
        fprintf(stderr, "%s(%a, %a): %a, expected %a (%llu ULP)\n",
                name, input, second, actual, expected, (unsigned long long)distance);
}

#define CHECK_FLOAT(name, oracle, x) compare(#name, x, 0, test_##name(x), (float)oracle((long double)(float)(x)), 1, 0)
#define CHECK_DOUBLE(name, oracle, x) compare(#name, x, 0, test_##name(x), (double)oracle((long double)(x)), 0, 0)

static void check_unary(double value) {
    CHECK_DOUBLE(exp, expl, value);
    CHECK_DOUBLE(log, logl, value);
    CHECK_DOUBLE(sin, sinl, value);
    CHECK_DOUBLE(cos, cosl, value);
    float single = (float)value;
    CHECK_FLOAT(expf, expl, single);
    CHECK_FLOAT(logf, logl, single);
    CHECK_FLOAT(expm1f, expm1l, single);
    CHECK_FLOAT(tanhf, tanhl, single);
    CHECK_FLOAT(erff, erfl, single);
    CHECK_FLOAT(sinf, sinl, single);
    CHECK_FLOAT(cosf, cosl, single);
}

static void check_binary(double left, double right, int exponent) {
    compare("fmod", left, right, test_fmod(left, right), fmod(left, right), 0, 1);
    compare("scalbn", left, exponent, test_scalbn(left, exponent), scalbn(left, exponent), 0, 1);
    int actual_power = 0, expected_power = 0;
    double actual = test_frexp(left, &actual_power), expected = frexp(left, &expected_power);
    compare("frexp", left, 0, actual, expected, 0, 1);
    if (isfinite(left) && actual_power != expected_power) failures++;
    float base = (float)left, power = (float)right;
    compare("powf", base, power, test_powf(base, power),
            (float)powl((long double)base, (long double)power), 1, 0);
}

int main(void) {
    static const double special[] = {
        0.0, -0.0, INFINITY, -INFINITY, NAN, -NAN,
        1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5,
        DBL_MAX, -DBL_MAX, DBL_MIN, -DBL_MIN, 0x1p-1074, -0x1p-1074,
        FLT_MAX, -FLT_MAX, FLT_MIN, -FLT_MIN, 0x1p-149, -0x1p-149,
        0x1p24, -0x1p24, 0x1.fffffep23, -0x1.fffffep23,
    };
    for (unsigned i = 0; i < sizeof(special) / sizeof(special[0]); i++) {
        check_unary(special[i]);
        for (unsigned j = 0; j < sizeof(special) / sizeof(special[0]); j++)
            check_binary(special[i], special[j], (j & 1) ? INT_MAX : INT_MIN);
    }
    /* Boundaries of both range reducers, the normal/subnormal transition,
     * exponent cutoffs, and logarithms just to either side of one. */
    static const double boundaries[] = {
        1, 4, 0.5, 0x1p20, 0x1p-1022, 0x1p-126, 88.722839, -103.972084,
        709.782712893384, -745.133219101941, -708.3964185322641,
    };
    for (unsigned i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); i++) {
        double left = boundaries[i], right = left;
        float left_f = (float)left, right_f = left_f;
        for (int j = 0; j < 128; j++) {
            check_unary(left); check_unary(right);
            check_unary(left_f); check_unary(right_f);
            left = nextafter(left, -INFINITY); right = nextafter(right, INFINITY);
            left_f = nextafterf(left_f, -INFINITY); right_f = nextafterf(right_f, INFINITY);
        }
    }
    long double half_pi = acosl(-1.0L) / 2;
    /* Consecutive multiples expose cancellation that powers of two do not:
     * multiplying a split constant by a non-power-of-two can itself round. */
    for (int multiple = 1; multiple <= 700000; multiple++) {
        double angle = (double)(multiple * half_pi);
        for (int neighbor = -1; neighbor <= 1; neighbor++) {
            double input = neighbor < 0 ? nextafter(angle, 0) :
                neighbor > 0 ? nextafter(angle, INFINITY) : angle;
            CHECK_DOUBLE(sin, sinl, input);
            CHECK_DOUBLE(cos, cosl, input);
        }
    }
    for (int power = 0; power < 1024; power++) {
        double angle = (double)scalbnl(half_pi, power);
        check_unary(angle);
        check_unary(nextafter(angle, 0));
        check_unary(nextafter(angle, INFINITY));
    }
    for (int i = 0; i < 100000; i++) {
        double left = (TestDouble){.bits = random_bits()}.value;
        double right = (TestDouble){.bits = random_bits()}.value;
        check_unary(left);
        check_binary(left, right, (int)(random_bits() % 4400) - 2200);
        check_unary((TestFloat){.bits = (uint32_t)random_bits()}.value);
        double unit = (double)(random_bits() >> 11) * 0x1p-53;
        CHECK_DOUBLE(exp, expl, unit * 1456 - 746);
        CHECK_DOUBLE(log, logl, unit * 1.5 + 0.5);
        CHECK_FLOAT(erff, erfl, (float)(unit * 8 - 4));
        CHECK_FLOAT(tanhf, tanhl, (float)(unit * 40 - 20));
        check_binary(unit * 2 - 1, unit * 200 - 100, i % 128 - 64);
    }
    printf("WASM scalar math: %u checks, %u failures; maximum %llu F32 / %llu F64 ULP\n",
           checks, failures, (unsigned long long)worst_float, (unsigned long long)worst_double);
    return failures ? 1 : 0;
}
