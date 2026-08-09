#ifndef VOLVOXAI_EXAMPLE_TINY_RECEIPT_SPLIT_W8A8_H
#define VOLVOXAI_EXAMPLE_TINY_RECEIPT_SPLIT_W8A8_H

#include <stddef.h>
#include <stdint.h>

#include "volvoxai.h"

/* Native host session for the qualified TinyReceipt encoder/decoder package. */
int tiny_receipt_split_w8a8_run(int argc, char** argv);

/* Exact strict-provider report predicate shared by the example and its
 * correctness tests. */
int tiny_receipt_split_w8a8_report_proves_strict_backend(
    const VxReport* report, const char* expected_backend);

/* Native example contract probe used by correctness tests. The prompt follows
 * the CLI contract: valid UTF-8, already normalized to NFC. */
int tiny_receipt_split_w8a8_tokenizer_roundtrip(
    const char* package_path, const char* prompt, int32_t* ids,
    size_t ids_capacity, size_t* ids_count, char* decoded,
    size_t decoded_capacity);

typedef struct TinyReceiptSplitShapeEvidence {
    int32_t shape_mode;
    int32_t question_length;
    int32_t target_length;
    int32_t memory_length;
    int32_t selected_family_id;
    int32_t generated_tokens;
    int32_t emitted_token_ids[191];
    uint64_t token_digest;
    uint64_t encoder_result_bytes;
    uint64_t decoder_seed_result_bytes;
    uint64_t decoder_step_result_bytes;
    double encoder_ms;
    double decoder_seed_ms;
    double decoder_warm_ms;
    char encoder_route[512];
    char decoder_seed_route[512];
    char decoder_step_route[512];
    int32_t decoder_seed_past_length;
    int32_t decoder_seed_present_length;
    int32_t decoder_step_past_length;
    int32_t decoder_step_present_length;
    int32_t explicit_kv_sentinel_preserved;
} TinyReceiptSplitShapeEvidence;

/* Correctness/benchmark probe: reuse one CPU encoder and decoder context while
 * alternating active requests and advance the manifest-declared explicit cache
 * tensors one token at a time. */
int tiny_receipt_split_w8a8_profile_sequence(
    const char* package_path, const char* image_path,
    const char* const* prompts, const int32_t* maximum_new_tokens,
    size_t request_count, TinyReceiptSplitShapeEvidence* evidence);

/* TinyReceipt qualification probe: reuse one strict backend runtime and its
 * encoder/decoder contexts across per-request active/maximum-padded modes. */
int tiny_receipt_split_w8a8_qualify_sequence(
    const char* package_path, const char* image_path,
    const char* const* prompts, const int32_t* maximum_new_tokens,
    const int32_t* shape_modes, size_t request_count, const char* backend,
    TinyReceiptSplitShapeEvidence* evidence);

#endif
