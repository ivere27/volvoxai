#ifndef VOLVOXAI_EXAMPLE_TINY_RECEIPT_SPLIT_W8A8_H
#define VOLVOXAI_EXAMPLE_TINY_RECEIPT_SPLIT_W8A8_H

#include <stddef.h>
#include <stdint.h>

/* Native host session for the qualified TinyReceipt encoder/decoder package. */
int tiny_receipt_split_w8a8_run(int argc, char** argv);

/* Native example contract probe used by correctness tests. The prompt follows
 * the CLI contract: valid UTF-8, already normalized to NFC. */
int tiny_receipt_split_w8a8_tokenizer_roundtrip(
    const char* package_path, const char* prompt, int32_t* ids,
    size_t ids_capacity, size_t* ids_count, char* decoded,
    size_t decoded_capacity);

#endif
