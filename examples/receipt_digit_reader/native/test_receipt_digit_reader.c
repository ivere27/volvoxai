/* Self-contained tests for the receipt digit reader's host contracts.
 *
 * No model data and no runtime: these cover the two places where an
 * application-side mistake silently produces a plausible wrong number --
 * preprocessing that disagrees with the producer, and slot decoding that
 * splices unrelated digits together.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "receipt_digit_reader.h"

static int failures = 0;

static void check(int condition, const char* label) {
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", label);
        failures++;
    }
}

static void check_text(const char* actual, const char* expected, const char* label) {
    if (strcmp(actual, expected) != 0) {
        fprintf(stderr, "FAIL %s: got '%s', expected '%s'\n", label, actual, expected);
        failures++;
    }
}

static ReceiptDigitLayout released_layout(void) {
    ReceiptDigitLayout layout;
    layout.slots = 16;
    layout.phone_slots = 12;
    layout.num_classes = 11;
    layout.blank_class = 10;
    return layout;
}

static void set_slot(float* logits, const ReceiptDigitLayout* layout, int slot, int class_index) {
    int candidate;
    for (candidate = 0; candidate < layout->num_classes; candidate++) {
        logits[slot * layout->num_classes + candidate] = candidate == class_index ? 1.0f : 0.0f;
    }
}

static void test_decode_reads_until_blank(void) {
    ReceiptDigitLayout layout = released_layout();
    float logits[16 * 11];
    ReceiptDigitRecord record;
    int slot;
    const int phone[7] = {0, 1, 2, 3, 4, 5, 6};
    const int street[3] = {9, 9, 9};
    for (slot = 0; slot < layout.slots; slot++) set_slot(logits, &layout, slot, layout.blank_class);
    for (slot = 0; slot < 7; slot++) set_slot(logits, &layout, slot, phone[slot]);
    for (slot = 0; slot < 3; slot++) set_slot(logits, &layout, 12 + slot, street[slot]);

    check(receipt_digit_decode(logits, &layout, &record) == 0, "decode accepts released layout");
    check_text(record.phone, "0123456", "phone digits");
    check_text(record.street, "999", "street digits");
}

static void test_decode_stops_at_an_interior_blank(void) {
    /* A blank in the middle means the number ended there. Dropping it instead
     * would splice the trailing digits on and produce a plausible wrong
     * number, which is indistinguishable from a right one downstream. */
    ReceiptDigitLayout layout = released_layout();
    float logits[16 * 11];
    ReceiptDigitRecord record;
    int slot;
    for (slot = 0; slot < layout.slots; slot++) set_slot(logits, &layout, slot, layout.blank_class);
    set_slot(logits, &layout, 0, 5);
    set_slot(logits, &layout, 1, 6);
    set_slot(logits, &layout, 2, layout.blank_class);
    set_slot(logits, &layout, 3, 7);

    check(receipt_digit_decode(logits, &layout, &record) == 0, "decode accepts interior blank");
    check_text(record.phone, "56", "phone stops at the first blank");
}

static void test_decode_rejects_an_invalid_layout(void) {
    ReceiptDigitLayout layout = released_layout();
    float logits[16 * 11] = {0};
    ReceiptDigitRecord record;
    layout.blank_class = layout.num_classes;
    check(receipt_digit_decode(logits, &layout, &record) != 0, "decode rejects an out-of-range blank");
    layout = released_layout();
    layout.slots = RECEIPT_DIGIT_MAX_SLOTS + 1;
    check(receipt_digit_decode(logits, &layout, &record) != 0, "decode rejects an oversized layout");
    layout = released_layout();
    check(receipt_digit_decode(NULL, &layout, &record) != 0, "decode rejects missing logits");
}

static void test_preprocess_normalizes_to_the_producer_range(void) {
    const unsigned char pixels[4] = {0, 255, 255, 0};
    float out[4];
    check(receipt_digit_preprocess(pixels, 2, 2, 1, out, 2, 2) == 0, "identity resize succeeds");
    check(fabsf(out[0] + 1.0f) < 1e-6f, "black maps to -1");
    check(fabsf(out[1] - 1.0f) < 1e-6f, "white maps to +1");
}

static void test_preprocess_uses_half_pixel_centers(void) {
    /* Downscaling 4->2 with half-pixel centers samples at source x = 0.5 and
     * 2.5, averaging neighbours. The corner convention would return the
     * endpoints instead, shifting thin digit strokes by up to half a pixel. */
    const unsigned char row[4] = {0, 85, 170, 255};
    float out[2];
    float expected_left = ((0.0f + 85.0f) / 2.0f / 255.0f - 0.5f) / 0.5f;
    float expected_right = ((170.0f + 255.0f) / 2.0f / 255.0f - 0.5f) / 0.5f;
    check(receipt_digit_preprocess(row, 4, 1, 1, out, 2, 1) == 0, "downscale succeeds");
    check(fabsf(out[0] - expected_left) < 1e-5f, "left sample averages its neighbours");
    check(fabsf(out[1] - expected_right) < 1e-5f, "right sample averages its neighbours");
}

static void test_preprocess_ignores_alpha(void) {
    /* Compositing would change values the model was trained and calibrated on. */
    const unsigned char rgba[8] = {10, 20, 30, 255, 10, 20, 30, 0};
    float out[2];
    check(receipt_digit_preprocess(rgba, 2, 1, 4, out, 2, 1) == 0, "rgba preprocess succeeds");
    check(fabsf(out[0] - out[1]) < 1e-6f, "alpha does not change the reduced value");
}

static void test_preprocess_rejects_unsupported_geometry(void) {
    const unsigned char pixels[2] = {0, 0};
    float out[2];
    check(receipt_digit_preprocess(pixels, 1, 1, 2, out, 1, 1) != 0, "rejects two channels");
    check(receipt_digit_preprocess(pixels, 0, 1, 1, out, 1, 1) != 0, "rejects an empty source");
    check(receipt_digit_preprocess(NULL, 1, 1, 1, out, 1, 1) != 0, "rejects missing pixels");
}

int main(void) {
    test_decode_reads_until_blank();
    test_decode_stops_at_an_interior_blank();
    test_decode_rejects_an_invalid_layout();
    test_preprocess_normalizes_to_the_producer_range();
    test_preprocess_uses_half_pixel_centers();
    test_preprocess_ignores_alpha();
    test_preprocess_rejects_unsupported_geometry();
    if (failures) {
        fprintf(stderr, "%d receipt digit reader check(s) failed\n", failures);
        return 1;
    }
    printf("receipt digit reader native checks passed\n");
    return 0;
}
