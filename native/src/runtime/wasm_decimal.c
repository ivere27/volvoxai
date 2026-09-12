/* Decimal conversion for the freestanding runtime, included by wasm_libc.c.
 *
 * Parsing rounds an integer ratio directly to binary64. Formatting expands
 * the exact dyadic value into decimal digits before rounding those digits.
 * Both use round-to-nearest, ties-to-even, with bounded stack storage.
 * No host conversion, floating approximation, or third-party implementation
 * supplies the conversion arithmetic. See wasm_numbers.md for the bounds.
 */

#define VX_DECIMAL_DIGITS 1400
#define VX_DECIMAL_WORDS 224

typedef struct {
    uint32_t words[VX_DECIMAL_WORDS];
    int used;
} VxDecimalInteger;

typedef union { double value; uint64_t bits; } VxDecimalDouble;

static void vx_decimal_set(VxDecimalInteger* a, uint64_t value) {
    a->words[0] = (uint32_t)value;
    a->words[1] = (uint32_t)(value >> 32);
    a->used = a->words[1] ? 2 : value ? 1 : 0;
}

static void vx_decimal_multiply(VxDecimalInteger* a, uint32_t factor,
                                uint32_t addend) {
    uint64_t carry = addend;
    for (int i = 0; i < a->used; i++) {
        carry += (uint64_t)a->words[i] * factor;
        a->words[i] = (uint32_t)carry;
        carry >>= 32;
    }
    if (carry) a->words[a->used++] = (uint32_t)carry;
}

static int vx_decimal_bits(const VxDecimalInteger* a) {
    return a->used ? (a->used - 1) * 32 +
        32 - __builtin_clz(a->words[a->used - 1]) : 0;
}

static void vx_decimal_shift(VxDecimalInteger* a, int shift) {
    if (!a->used || !shift) return;
    int whole = shift / 32, part = shift % 32;
    uint32_t carry = 0;
    if (part) {
        for (int i = 0; i < a->used; i++) {
            uint32_t word = a->words[i];
            a->words[i] = (word << part) | carry;
            carry = word >> (32 - part);
        }
        if (carry) a->words[a->used++] = carry;
    }
    for (int i = a->used - 1; i >= 0; i--) a->words[i + whole] = a->words[i];
    for (int i = 0; i < whole; i++) a->words[i] = 0;
    a->used += whole;
}

static uint32_t vx_decimal_shifted_word(const VxDecimalInteger* a,
                                        int shift, int index) {
    int source = index - shift / 32, part = shift % 32;
    uint32_t word = source >= 0 && source < a->used ? a->words[source] << part : 0;
    if (part && source > 0 && source <= a->used)
        word |= a->words[source - 1] >> (32 - part);
    return word;
}

static int vx_decimal_compare_shift(const VxDecimalInteger* a,
                                     const VxDecimalInteger* b, int shift) {
    int a_bits = vx_decimal_bits(a), b_bits = vx_decimal_bits(b) + shift;
    if (a_bits != b_bits) return a_bits > b_bits ? 1 : -1;
    for (int i = a->used - 1; i >= 0; i--) {
        uint32_t word = vx_decimal_shifted_word(b, shift, i);
        if (a->words[i] != word) return a->words[i] > word ? 1 : -1;
    }
    return 0;
}

static void vx_decimal_subtract_shift(VxDecimalInteger* a,
                                      const VxDecimalInteger* b, int shift) {
    uint64_t borrow = 0;
    for (int i = 0; i < a->used; i++) {
        uint64_t sub = (uint64_t)vx_decimal_shifted_word(b, shift, i) + borrow;
        uint32_t original = a->words[i];
        a->words[i] = original - (uint32_t)sub;
        borrow = (uint64_t)original < sub;
    }
    while (a->used && !a->words[a->used - 1]) a->used--;
}

static uint32_t vx_decimal_divide_small(VxDecimalInteger* a, uint32_t divisor) {
    uint64_t remainder = 0;
    for (int i = a->used - 1; i >= 0; i--) {
        uint64_t dividend = (remainder << 32) | a->words[i];
        a->words[i] = (uint32_t)(dividend / divisor);
        remainder = dividend % divisor;
    }
    while (a->used && !a->words[a->used - 1]) a->used--;
    return (uint32_t)remainder;
}

static double vx_decimal_parse(const char* text, size_t length) {
    VxDecimalInteger numerator, denominator, compared;
    VxDecimalDouble result;
    size_t at = 0;
    int negative = 0, fraction = 0, sticky = 0, kept = 0;
    int64_t significant = 0, fractional = 0, explicit_exponent = 0;
    if (at < length && (text[at] == '+' || text[at] == '-'))
        negative = text[at++] == '-';
    vx_decimal_set(&numerator, 0);
    while (at < length && text[at] != 'e' && text[at] != 'E') {
        unsigned char ch = (unsigned char)text[at++];
        if (ch == '.') { fraction = 1; continue; }
        if (fraction) fractional++;
        if (!significant && ch == '0') continue;
        significant++;
        if (kept < VX_DECIMAL_DIGITS) {
            vx_decimal_multiply(&numerator, 10, ch - '0');
            kept++;
        } else if (ch != '0') sticky = 1;
    }
    if (at < length) {
        int exponent_negative = 0;
        at++;
        if (at < length && (text[at] == '+' || text[at] == '-'))
            exponent_negative = text[at++] == '-';
        for (; at < length; at++) {
            if (explicit_exponent < INT64_C(10000000000))
                explicit_exponent = explicit_exponent * 10 + text[at] - '0';
        }
        if (exponent_negative) explicit_exponent = -explicit_exponent;
    }
    result.bits = (uint64_t)negative << 63;
    if (!significant) return result.value;
    int64_t order = significant - fractional + explicit_exponent - 1;
    if (order > 308) { result.bits |= UINT64_C(0x7ff0000000000000); return result.value; }
    if (order < -324) return result.value;
    int exponent10 = (int)(order + 1 - kept);

    /* The operands in this fast path are exact binary64 integers. One
     * hardware multiply/divide therefore performs the required rounding. */
    if (significant <= 15 && exponent10 >= -22 && exponent10 <= 22) {
        static const double powers[] = {
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
            1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17,
            1e18, 1e19, 1e20, 1e21, 1e22
        };
        uint64_t integer = numerator.words[0];
        if (numerator.used > 1) integer |= (uint64_t)numerator.words[1] << 32;
        double value = exponent10 < 0 ? (double)integer / powers[-exponent10] :
            (double)integer * powers[exponent10];
        return negative ? -value : value;
    }

    vx_decimal_set(&denominator, 1);
    for (int i = 0; i < exponent10; i++) vx_decimal_multiply(&numerator, 10, 0);
    for (int i = 0; i < -exponent10; i++) vx_decimal_multiply(&denominator, 10, 0);
    int exponent2 = vx_decimal_bits(&numerator) - vx_decimal_bits(&denominator);
    if (exponent2 >= 0) {
        if (vx_decimal_compare_shift(&numerator, &denominator, exponent2) < 0) exponent2--;
    } else {
        compared = numerator;
        vx_decimal_shift(&compared, -exponent2);
        if (vx_decimal_compare_shift(&compared, &denominator, 0) < 0) exponent2--;
    }
    int shift = exponent2 < -1022 ? 1074 : 52 - exponent2;
    if (shift >= 0) vx_decimal_shift(&numerator, shift);
    else vx_decimal_shift(&denominator, -shift);
    uint64_t quotient = 0;
    for (int bit = vx_decimal_bits(&numerator) - vx_decimal_bits(&denominator);
         bit >= 0; bit--) {
        if (vx_decimal_compare_shift(&numerator, &denominator, bit) >= 0) {
            vx_decimal_subtract_shift(&numerator, &denominator, bit);
            quotient |= UINT64_C(1) << bit;
        }
    }
    vx_decimal_shift(&numerator, 1);
    int rounding = vx_decimal_compare_shift(&numerator, &denominator, 0);
    if (rounding > 0 || (!rounding && (sticky || (quotient & 1)))) quotient++;
    if (exponent2 < -1022) result.bits |= quotient;
    else {
        if (quotient == (UINT64_C(1) << 53)) { quotient >>= 1; exponent2++; }
        if (exponent2 > 1023) result.bits |= UINT64_C(0x7ff0000000000000);
        else result.bits |= (uint64_t)(exponent2 + 1023) << 52 |
            (quotient & UINT64_C(0x000fffffffffffff));
    }
    return result.value;
}

typedef struct {
    char digits[VX_DECIMAL_DIGITS];
    int count;
    int exponent;
} VxDecimalDigits;

static void vx_decimal_expand(VxDecimalDigits* out, uint64_t bits) {
    uint64_t mantissa = bits & UINT64_C(0x000fffffffffffff);
    int binary_exponent = (int)((bits >> 52) & 2047);
    if (binary_exponent) mantissa |= UINT64_C(1) << 52;
    binary_exponent = binary_exponent ? binary_exponent - 1075 : -1074;
    if (!mantissa) { out->digits[0] = '0'; out->count = 1; out->exponent = 0; return; }
    while (binary_exponent < 0 && !(mantissa & 1)) { mantissa >>= 1; binary_exponent++; }
    VxDecimalInteger integer;
    vx_decimal_set(&integer, mantissa);
    int scale = binary_exponent < 0 ? -binary_exponent : 0;
    if (binary_exponent > 0) vx_decimal_shift(&integer, binary_exponent);
    for (int i = 0; i < scale; i++) vx_decimal_multiply(&integer, 5, 0);
    out->count = 0;
    do {
        uint32_t block = vx_decimal_divide_small(&integer, 1000000000);
        int digits = integer.used ? 9 : 1;
        if (!integer.used) for (uint32_t remaining = block; remaining >= 10; remaining /= 10) digits++;
        for (int i = 0; i < digits; i++) {
            out->digits[out->count++] = (char)('0' + block % 10);
            block /= 10;
        }
    } while (integer.used);
    for (int i = 0; i < out->count / 2; i++) {
        char swap = out->digits[i];
        out->digits[i] = out->digits[out->count - i - 1];
        out->digits[out->count - i - 1] = swap;
    }
    out->exponent = out->count - scale - 1;
}

static void vx_decimal_round(VxDecimalDigits* value, int64_t keep) {
    if (keep < value->count) {
        int up = 0;
        if (keep >= 0) {
            int tail = 0;
            for (int i = (int)keep + 1; i < value->count; i++) tail |= value->digits[i] != '0';
            up = value->digits[keep] > '5' || (value->digits[keep] == '5' &&
                (tail || (keep > 0 && ((value->digits[keep - 1] - '0') & 1))));
        }
        if (keep <= 0) {
            value->digits[0] = up ? '1' : '0';
            value->exponent = up ? value->exponent + 1 : 0;
            value->count = 1;
        } else {
            value->count = (int)keep;
            int index = value->count - 1;
            while (up && index >= 0 && value->digits[index] == '9') value->digits[index--] = '0';
            if (up && index >= 0) value->digits[index]++;
            else if (up) { value->digits[0] = '1'; value->exponent++; }
        }
    }
    while (value->count > 1 && value->digits[value->count - 1] == '0') value->count--;
}

typedef struct { char* bytes; size_t capacity; int64_t length; } VxDecimalWriter;

static void vx_decimal_repeat(VxDecimalWriter* out, char ch, int64_t count) {
    if (count <= 0) return;
    size_t available = out->capacity && out->length < (int64_t)(out->capacity - 1) ?
        out->capacity - 1 - (size_t)out->length : 0;
    size_t copied = (uint64_t)count < available ? (size_t)count : available;
    for (size_t i = 0; i < copied; i++) out->bytes[(size_t)out->length + i] = ch;
    out->length += count;
}

static void vx_decimal_write(VxDecimalWriter* out, const char* text, int count) {
    for (int i = 0; i < count; i++) vx_decimal_repeat(out, text[i], 1);
}

static int vx_decimal_format(char* bytes, size_t capacity, double number,
                              int precision, int width, int zero_pad, char specifier) {
    VxDecimalDouble input = { .value = number };
    VxDecimalWriter out = {bytes, capacity, 0};
    VxDecimalDigits value;
    int negative = (int)(input.bits >> 63), uppercase = specifier < 'a';
    char format = (char)(specifier | 32);
    uint64_t magnitude = input.bits & UINT64_C(0x7fffffffffffffff);
    if (magnitude >= UINT64_C(0x7ff0000000000000)) {
        const char* text = magnitude == UINT64_C(0x7ff0000000000000) ?
            (uppercase ? "INF" : "inf") : (uppercase ? "NAN" : "nan");
        vx_decimal_repeat(&out, ' ', (int64_t)width - negative - 3);
        if (negative) vx_decimal_repeat(&out, '-', 1);
        vx_decimal_write(&out, text, 3);
    } else {
        if (precision < 0) precision = 6;
        if (format == 'g' && !precision) precision = 1;
        vx_decimal_expand(&value, magnitude);
        int64_t keep = format == 'f' ? (int64_t)value.exponent + 1 + precision :
            (int64_t)precision + (format == 'e');
        vx_decimal_round(&value, keep);
        int scientific = format == 'e' || (format == 'g' &&
            (value.exponent < -4 || value.exponent >= precision));
        int64_t fractional = format == 'g' ? (scientific ? value.count - 1 :
            value.count - 1 - value.exponent) : precision;
        if (fractional < 0) fractional = 0;
        int exponent_digits = value.exponent <= -100 || value.exponent >= 100 ? 3 : 2;
        int64_t integer_digits = scientific || value.exponent < 0 ? 1 : value.exponent + 1;
        int64_t body = integer_digits + (fractional ? 1 + fractional : 0) +
            (scientific ? 2 + exponent_digits : 0);
        int64_t padding = (int64_t)width - body - negative;
        if (!zero_pad) vx_decimal_repeat(&out, ' ', padding);
        if (negative) vx_decimal_repeat(&out, '-', 1);
        if (zero_pad) vx_decimal_repeat(&out, '0', padding);
        if (scientific) vx_decimal_write(&out, value.digits, 1);
        else if (value.exponent < 0) vx_decimal_repeat(&out, '0', 1);
        else {
            int copied = value.count < integer_digits ? value.count : (int)integer_digits;
            vx_decimal_write(&out, value.digits, copied);
            vx_decimal_repeat(&out, '0', integer_digits - copied);
        }
        if (fractional) {
            vx_decimal_repeat(&out, '.', 1);
            int start = scientific ? 1 : value.exponent + 1;
            if (start < 0) {
                int64_t zeros = fractional < -start ? fractional : -start;
                vx_decimal_repeat(&out, '0', zeros);
                fractional -= zeros;
                start = 0;
            }
            int copied = start < value.count ? value.count - start : 0;
            if (copied > fractional) copied = (int)fractional;
            if (copied) vx_decimal_write(&out, value.digits + start, copied);
            vx_decimal_repeat(&out, '0', fractional - copied);
        }
        if (scientific) {
            vx_decimal_repeat(&out, uppercase ? 'E' : 'e', 1);
            vx_decimal_repeat(&out, value.exponent < 0 ? '-' : '+', 1);
            int exponent = value.exponent < 0 ? -value.exponent : value.exponent;
            if (exponent_digits == 3) vx_decimal_repeat(&out, (char)('0' + exponent / 100), 1);
            vx_decimal_repeat(&out, (char)('0' + exponent / 10 % 10), 1);
            vx_decimal_repeat(&out, (char)('0' + exponent % 10), 1);
        }
    }
    if (capacity) bytes[out.length < (int64_t)capacity ? (size_t)out.length : capacity - 1] = '\0';
    return out.length > INT_MAX ? -1 : (int)out.length;
}

#undef VX_DECIMAL_DIGITS
#undef VX_DECIMAL_WORDS
