# Freestanding numbers

The WASM runtime supplies the numeric functions its C callers need in
`wasm_decimal.c` and `wasm_math.c`. These are private implementation files;
`proto/volvoxai.proto` remains the public API. Native releases use the platform's
libc and libm. Neither WASM file delegates arithmetic to JavaScript.

The implementations are written from integer arithmetic and mathematical
series, without incorporating another library's source or coefficient tables.
`tools/generate_wasm_math.py` derives its constants from definitions: Machin's
identity for pi, the logarithm of two, square roots, and rational factorials.
Generation at 500 and 600 decimal digits must produce identical output. Edit
the generator, then regenerate `wasm_math_constants.h`.

## Decimal conversion

`wasm_libc.c` scans decimal prefixes, including an optional sign, decimal point
and exponent. It preserves the existing decimal-only lexical contract; this is
not a complete locale-aware C library. Parsing converts the decimal to an exact
integer ratio, divides for a binary64 significand, and rounds the remainder to
nearest with ties to even. Signed zero and overflow/underflow are preserved.
At most 15 significant digits and decimal exponents in [-22, 22] use one
floating multiply/divide on exactly represented operands.

Long input retains the first 1400 significant digits and a flag recording any
discarded nonzero digit. Every binary64 rounding midpoint is a dyadic rational
with at most 1075 decimal fractional positions and 309 integer positions.
Thus its significant expansion fits within the retained prefix; a discarded
tail only changes a tie. Input length and exponent accounting use 64-bit
integers. The explicit exponent saturates beyond the maximum WASM input length,
so a huge exponent cannot wrap into the representable range.

Each integer has 224 32-bit words (7168 bits). Orders outside [-324, 308] are
classified before building powers of ten. Within that range, a retained
1400-digit numerator is under 4651 bits and the denominator is at most
10^1723, under 5724 bits. Binary normalization and remainder comparison fit
below 5800 bits. No allocation grows with the input length.

Formatting expands the exact dyadic value into decimal digits, then applies
decimal ties-to-even rounding for `%g/%G`, `%e/%E`, or `%f/%F`. Width, zero
padding, precision and truncated output retain the caller's formatting
contract. The bounded writer reports the untruncated length without storing
the whole padded result. Precision does not enlarge the stack arrays.

## Scalar math

The implementation assumes IEEE binary32/binary64, round-to-nearest with ties
to even, and no fast-math or fused expression contraction.

- `scalbn`, `frexp` and `fmod` use exact exponent/significand operations.
- `exp` reduces by a split ln(2), evaluates its series through degree 20 and
  scales by a power of two. `expm1f` retains small differences directly.
- `log` normalizes into [sqrt(1/2), sqrt(2)], then evaluates
  `2 * (z + z^3/3 + ... + z^39/39)`, where `z = (x-1)/(x+1)`.
- `sin` and `cos` reduce to [-pi/4, pi/4] and evaluate series through degrees
  19 and 20. Small arguments use three split pi/2 components. Large arguments
  and remainders below 2^-20 use the exact significand multiplied by a 1280-bit
  fixed-point reciprocal of pi/2. This avoids cancellation near multiples;
  even at DBL_MAX, reciprocal truncation changes the reduced fraction by less
  than 2^-256. There is no integer cast of an unreduced large argument.
- Float functions evaluate in double before the final float rounding. `powf`
  handles domain and sign rules before applying exp/log. `tanhf` uses expm1.
  `erff` uses a positive-term Gaussian integral expansion after factoring out
  exp(-x*x), avoiding cancellation; at |x| >= 4 its binary32 result rounds to 1.

These are approximations, not a claim of correctly rounded transcendental
functions for every input. The regression contract allows 1 ULP for binary32
and 4 ULP for binary64 transcendental results. The exact functions, signs of
zero, and infinity/NaN classification are checked separately.

## Validation

`test_wasm_math` compares millions of calls against the platform's long-double
math functions rounded to the destination precision. Inputs cover random bit
patterns, domain boundaries, consecutive pi/2 multiples and their neighbors,
huge angles, normal/subnormal transitions and
overflow/underflow. `test_wasm_libc_format` compares decimal parsing and printing
against the platform library, including exact midpoints with tails beyond 1400
digits, random round trips, all supported float formats and bounded buffers.
Both tests carry the `native-invariant` CTest label.

Run `python3 tools/generate_wasm_math.py --check` for constant freshness. Changes
must also pass inference/full WASM builds, the proto API and authoring/training
suites, which exercise the compiled implementation through the public API.
