#ifndef NNAPI_ENGINE_H
#define NNAPI_ENGINE_H

#include <stddef.h>
#include <stdint.h>

enum { NNAPI_MODEL_CACHE_CAPACITY = 16 };

int nnapi_init(void);
void nnapi_cleanup(void);
/* Weight/bias storage must remain immutable for the context lifetime. Returns
 * one only after NNAPI completed and the staged result was committed to out.
 * A zero return leaves caller-owned output bytes untouched. */
int nnapi_matmul(const float* in, const float* w, const float* b, float* out,
                 int seq, int d_in, int d_out);
void nnapi_free_weight_cache(void);

typedef struct {
    size_t entry_count;
    size_t entry_capacity;
    size_t output_staging_capacity;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t model_builds;
    uint64_t model_build_failures;
    uint64_t evictions;
    uint64_t executions;
    uint64_t execution_failures;
    uint64_t limit_rejections;
    uint64_t missing_input_rejections;
    double last_model_build_time_ms;
    double total_model_build_time_ms;
    double cache_hit_rate;
} NnapiCacheTelemetry;
int nnapi_cache_telemetry(NnapiCacheTelemetry* telemetry);

#if defined(VOLVOXAI_NNAPI_TESTING)
int nnapi_test_cache_telemetry(NnapiCacheTelemetry* telemetry);
#endif

#endif
