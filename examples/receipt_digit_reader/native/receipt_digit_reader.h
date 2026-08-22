#ifndef VOLVOXAI_EXAMPLE_RECEIPT_DIGIT_READER_H
#define VOLVOXAI_EXAMPLE_RECEIPT_DIGIT_READER_H

#include <stddef.h>

/* The record the graph reads: two left-aligned digit strings. Both are sized
 * for the released 12 phone slots and 4 street slots plus a terminator; the
 * decoder refuses a package that declares more. */
#define RECEIPT_DIGIT_MAX_SLOTS 32

typedef struct ReceiptDigitRecord {
    char phone[RECEIPT_DIGIT_MAX_SLOTS + 1];
    char street[RECEIPT_DIGIT_MAX_SLOTS + 1];
} ReceiptDigitRecord;

/* Decode layout, read from the package manifest rather than assumed. */
typedef struct ReceiptDigitLayout {
    int slots;
    int phone_slots;
    int num_classes;
    int blank_class;
} ReceiptDigitLayout;

/* Grayscale + bilinear resize + symmetric normalization, matching the
 * producer's `(gray / 255 - 0.5) / 0.5` on a Pillow bilinear resize. `channels`
 * must be 1, 3, or 4; an alpha channel is ignored rather than composited. */
int receipt_digit_preprocess(const unsigned char* pixels,
                             int source_width,
                             int source_height,
                             int channels,
                             float* destination,
                             int target_width,
                             int target_height);

/* Decode one [slots, num_classes] logits block into a record. */
int receipt_digit_decode(const float* logits,
                         const ReceiptDigitLayout* layout,
                         ReceiptDigitRecord* record);

/* Decode a PNG/JPEG file straight into the network plane. */
int receipt_digit_load_image(const char* path,
                             float* destination,
                             int target_width,
                             int target_height,
                             char* error,
                             size_t error_size);

/* Read `manifest.json` from a published package directory. */
int receipt_digit_read_manifest(const char* package_directory,
                                ReceiptDigitLayout* layout,
                                int* width,
                                int* height,
                                char* input_name,
                                size_t input_name_size,
                                char* error,
                                size_t error_size);

/* Run one receipt through a published package on one strictly required
 * backend. `backend` is a VolvoxAI runtime backend identity such as "cpu",
 * "vulkan", "opengl", or "cuda". Note this is the runtime identity, not the
 * exporter target name ("native-cpu"). */
int receipt_digit_read_receipt(const char* package_directory,
                               const char* image_path,
                               const char* backend,
                               ReceiptDigitRecord* record,
                               char* error,
                               size_t error_size);

/* A retained session over one compiled package.
 *
 * `receipt_digit_read_receipt` is the whole lifecycle in one call, which is
 * the right shape for reading one receipt but the wrong shape for measuring
 * one: it would fold model load, compilation, and context creation into every
 * sample. A benchmark holds the session open and times only
 * `receipt_digit_session_execute`. */
typedef struct ReceiptDigitSession ReceiptDigitSession;

ReceiptDigitSession* receipt_digit_session_open(const char* package_directory,
                                                const char* backend,
                                                char* error,
                                                size_t error_size);

/* Declared network geometry, so a caller can size its own input plane. */
void receipt_digit_session_geometry(const ReceiptDigitSession* session,
                                    int* width,
                                    int* height,
                                    ReceiptDigitLayout* layout);

/* Execute one already-preprocessed plane. `elapsed_ms`, when non-NULL,
 * receives the execution and output-read time only: shape binding, dispatch,
 * synchronization, and the owned output snapshot. */
int receipt_digit_session_execute(ReceiptDigitSession* session,
                                  const float* plane,
                                  ReceiptDigitRecord* record,
                                  double* elapsed_ms,
                                  char* error,
                                  size_t error_size);

void receipt_digit_session_close(ReceiptDigitSession* session);

#endif
