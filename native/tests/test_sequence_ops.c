#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sequence_ops.h"

static int closef(float left, float right) {
    return fabsf(left - right) <= 2.0e-5f * fmaxf(1.0f, fabsf(right));
}

int main(void) {
    const float pi = 3.14159265358979323846f;
    float unary_input[] = {0.0f, pi * 0.5f, -pi, INFINITY};
    float sine[4], cosine[4];
    sin_f32(unary_input, sine, 4);
    cos_f32(unary_input, cosine, 4);
    assert(closef(sine[0], 0.0f) && closef(sine[1], 1.0f) && closef(sine[2], 0.0f));
    assert(closef(cosine[0], 1.0f) && closef(cosine[1], 0.0f) && closef(cosine[2], -1.0f));
    assert(isnan(sine[3]) && isnan(cosine[3]));

    {
        float input[] = {1, 2, 3, 4, 9, 10, -1, -2, -3, -4, 11, 12};
        float output[12];
        int32_t positions[] = {0, 2};
        unsigned char unaligned_positions[sizeof(positions) + 1];
        memcpy(unaligned_positions + 1, positions, sizeof(positions));
        assert(rope_f32(input, NULL, output, 1, 2, 6, 4, 10.0f, 1, 0, 0));
        assert(closef(output[0], input[0] * cosf(1.0f) - input[2] * sinf(1.0f)));
        assert(output[4] == 9.0f && output[5] == 10.0f);
        assert(rope_f32(input, positions, output, 1, 2, 6, 4, 10.0f, 0, 1, 1));
        assert(output[0] == 1.0f && output[1] == 2.0f);
        assert(closef(output[6], input[6] * cosf(2.0f) - input[7] * sinf(2.0f)));
        assert(rope_f32(input, unaligned_positions + 1, output,
                        1, 2, 6, 4, 10.0f, 0, 1, 1));
        assert(closef(output[6], input[6] * cosf(2.0f) - input[7] * sinf(2.0f)));
        assert(!rope_f32(input, positions, output, 1, 2, 6, 3, 10.0f, 0, 0, 1));
    }

    {
        float input[] = {1.0f, 2.0f};
        float delta[] = {0.5f, 0.25f};
        float a[] = {-1.0f};
        float b[] = {2.0f, 3.0f};
        float c[] = {0.5f};
        float initial[] = {0.25f};
        float output[2], final_state[1], scratch[1];
        float state0 = expf(-0.5f) * 0.25f + 0.5f * 2.0f * 1.0f;
        float state1 = expf(-0.25f) * state0 + 0.25f * 3.0f * 2.0f;
        assert(ssm_scan_f32(input, delta, a, b, c, NULL, NULL, initial,
                            output, final_state, scratch, 1, 2, 1, 1, 1, 0, 0));
        assert(closef(output[0], state0 * 0.5f));
        assert(closef(output[1], state1 * 0.5f));
        assert(closef(final_state[0], state1));
    }

    puts("sequence operator kernel tests passed");
    return 0;
}
