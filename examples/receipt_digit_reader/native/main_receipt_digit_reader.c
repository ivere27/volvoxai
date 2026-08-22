/* Receipt digit reader CLI.
 *
 * Reads one receipt image with a published VolvoxAI package on one strictly
 * required backend and prints the decoded record as JSON.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "receipt_digit_reader.h"

static int usage(int status) {
    FILE* stream = status == 0 ? stdout : stderr;
    fprintf(stream,
            "usage: receipt_digit_reader --package DIR --image FILE [--backend NAME]\n"
            "                            [--repeat N] [--warmup N]\n"
            "\n"
            "  --package DIR    published volvoxai-receipt-digit-reader package\n"
            "  --image FILE     PNG or JPEG receipt\n"
            "  --backend NAME   required backend identity (default: cpu).\n"
            "                   Accepts any identity the build registered, such as\n"
            "                   cpu, vulkan, opengl, or cuda. Selection is\n"
            "                   strict: an unavailable backend fails rather than\n"
            "                   falling back.\n"
            "  --repeat N       time N measured executions on one retained session\n"
            "                   and print their milliseconds. Model load,\n"
            "                   compilation, context creation, image decoding, and\n"
            "                   process startup are excluded.\n"
            "  --warmup N       untimed executions before measurement (default 3)\n");
    return status;
}

static int compare_double(const void* left, const void* right) {
    const double a = *(const double*)left;
    const double b = *(const double*)right;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static double median_of(double* values, int count) {
    qsort(values, (size_t)count, sizeof(double), compare_double);
    return count % 2 ? values[count / 2]
                     : (values[count / 2 - 1] + values[count / 2]) / 2.0;
}

static int benchmark(const char* package, const char* image, const char* backend,
                     int repeat, int warmup) {
    ReceiptDigitSession* session;
    ReceiptDigitRecord record;
    ReceiptDigitRecord first;
    char error[512] = {0};
    float* plane = NULL;
    double* samples = NULL;
    double elapsed = 0.0;
    double median = 0.0;
    int width = 0;
    int height = 0;
    int index;
    int status = 1;

    session = receipt_digit_session_open(package, backend, error, sizeof(error));
    if (!session) {
        fprintf(stderr, "[receipt_digit_reader] %s\n", error);
        return 1;
    }
    receipt_digit_session_geometry(session, &width, &height, NULL);
    plane = (float*)malloc((size_t)width * (size_t)height * sizeof(float));
    samples = (double*)malloc((size_t)repeat * sizeof(double));
    if (!plane || !samples) {
        fprintf(stderr, "[receipt_digit_reader] benchmark buffers could not be allocated\n");
        goto done;
    }
    if (receipt_digit_load_image(image, plane, width, height, error, sizeof(error)) != 0) {
        fprintf(stderr, "[receipt_digit_reader] %s\n", error);
        goto done;
    }
    for (index = 0; index < warmup; index++) {
        if (receipt_digit_session_execute(session, plane, &record, NULL,
                                          error, sizeof(error)) != 0) {
            fprintf(stderr, "[receipt_digit_reader] warmup %s\n", error);
            goto done;
        }
    }
    for (index = 0; index < repeat; index++) {
        if (receipt_digit_session_execute(session, plane, &record, &elapsed,
                                          error, sizeof(error)) != 0) {
            fprintf(stderr, "[receipt_digit_reader] %s\n", error);
            goto done;
        }
        /* A route that stops producing the same record is not a benchmark
         * result; timing a wrong answer measures nothing worth comparing. */
        if (index == 0) first = record;
        else if (strcmp(record.phone, first.phone) != 0 ||
                 strcmp(record.street, first.street) != 0) {
            fprintf(stderr, "[receipt_digit_reader] record changed between runs\n");
            goto done;
        }
        samples[index] = elapsed;
    }
    /* Evaluate the in-place sort before reading its endpoints. Function
     * argument evaluation order is unspecified in C, so doing all three in
     * one printf can observe one endpoint before qsort and the other after. */
    median = median_of(samples, repeat);
    printf("{\"backend\":\"%s\",\"phone\":\"%s\",\"street\":\"%s\",\"runs\":%d,"
           "\"median_ms\":%.4f,\"min_ms\":%.4f,\"max_ms\":%.4f,\"samples\":[",
           backend, first.phone, first.street, repeat,
           median, samples[0], samples[repeat - 1]);
    for (index = 0; index < repeat; index++) {
        printf("%s%.4f", index ? "," : "", samples[index]);
    }
    printf("]}\n");
    status = 0;

done:
    free(plane);
    free(samples);
    receipt_digit_session_close(session);
    return status;
}

int main(int argc, char** argv) {
    const char* package = NULL;
    const char* image = NULL;
    const char* backend = "cpu";
    ReceiptDigitRecord record;
    char error[512] = {0};
    int repeat = 0;
    int warmup = 3;
    int index;

    for (index = 1; index < argc; index++) {
        const char* flag = argv[index];
        if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) return usage(0);
        if (index + 1 >= argc) {
            fprintf(stderr, "[receipt_digit_reader] %s needs a value\n", flag);
            return usage(2);
        }
        if (strcmp(flag, "--package") == 0) package = argv[++index];
        else if (strcmp(flag, "--image") == 0) image = argv[++index];
        else if (strcmp(flag, "--backend") == 0) backend = argv[++index];
        else if (strcmp(flag, "--repeat") == 0) repeat = atoi(argv[++index]);
        else if (strcmp(flag, "--warmup") == 0) warmup = atoi(argv[++index]);
        else {
            fprintf(stderr, "[receipt_digit_reader] unknown option %s\n", flag);
            return usage(2);
        }
    }
    if (!package || !image) return usage(2);
    if (repeat < 0 || warmup < 0) {
        fprintf(stderr, "[receipt_digit_reader] --repeat and --warmup must be non-negative\n");
        return usage(2);
    }
    if (repeat > 0) return benchmark(package, image, backend, repeat, warmup);

    if (receipt_digit_read_receipt(package, image, backend, &record,
                                   error, sizeof(error)) != 0) {
        fprintf(stderr, "[receipt_digit_reader] %s\n",
                error[0] ? error : "the receipt could not be read");
        return 1;
    }
    printf("{\"phone\":\"%s\",\"street\":\"%s\",\"backend\":\"%s\"}\n",
           record.phone, record.street, backend);
    return 0;
}
