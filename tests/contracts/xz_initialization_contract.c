/* Exercise the production metadata and shader decoders together. The linker
 * wrapper counts real CRC initializations; it does not replace the decoder. */
#include "vx_api_convert.h"
#include "vx_api_contract.inc"
#include "shader_store.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#if !defined(VOLVOXAI_NO_THREADS)
#include <pthread.h>
#endif

static atomic_uint crc_initializations;

void __real_xz_crc32_init(void);
void __wrap_xz_crc32_init(void) {
    atomic_fetch_add(&crc_initializations, 1u);
    __real_xz_crc32_init();
}

static int describe_contract(void) {
    VolvoxaiV1DescribeApiRequest request;
    VolvoxaiV1ApiDescription response;
    volvoxai_v1_describe_api_request_init(&request);
    volvoxai_v1_api_description_init(&response);
    request.field_include_types = 1;
    request.field_service.data = (uint8_t*)"VxInferenceService";
    request.field_service.len = strlen("VxInferenceService");
    request.field_method.data = (uint8_t*)"LoadModel";
    request.field_method.len = strlen("LoadModel");
    int result = vx_api_describe_api(&request, &response, NULL);
    int valid = result == 0 && response.field_report &&
        response.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK &&
        response.field_methods.len == 1 && response.field_messages.len > 0 &&
        response.field_schema_sha256.len == 64;
    volvoxai_v1_api_description_free(&response);
    return valid;
}

static int decode_shader(void) {
    unsigned char* output = malloc(VX_API_CONTRACT_UNCOMPRESSED_SIZE);
    if (!output) return 0;
    /* The shader decoder accepts arbitrary bounded CRC32 XZ blocks. Use the
     * metadata block so the fixture needs neither a GPU nor generated shaders. */
    int valid = volvoxai_shader_decode_xz(
        vx_api_contract_payload_xz, sizeof(vx_api_contract_payload_xz),
        output, VX_API_CONTRACT_UNCOMPRESSED_SIZE) == VOLVOXAI_SHADER_STORE_OK;
    free(output);
    return valid;
}

#if !defined(VOLVOXAI_NO_THREADS)
static pthread_barrier_t start;

static void* decode_worker(void* opaque) {
    unsigned which = (unsigned)(uintptr_t)opaque;
    pthread_barrier_wait(&start);
    for (unsigned iteration = 0; iteration < 4; iteration++) {
        if (!(which % 2 ? describe_contract() : decode_shader()))
            return (void*)(uintptr_t)1;
    }
    return NULL;
}
#endif

int main(void) {
#if defined(VOLVOXAI_NO_THREADS)
    for (unsigned iteration = 0; iteration < 4; iteration++)
        if (!describe_contract() || !decode_shader()) return 1;
#else
    pthread_t threads[16];
    if (pthread_barrier_init(&start, NULL, 16) != 0) return 2;
    for (unsigned i = 0; i < 16; i++)
        if (pthread_create(&threads[i], NULL, decode_worker, (void*)(uintptr_t)i) != 0)
            return 3;
    for (unsigned i = 0; i < 16; i++) {
        void* result = NULL;
        if (pthread_join(threads[i], &result) != 0 || result) return 4;
    }
    pthread_barrier_destroy(&start);
#endif
    unsigned count = atomic_load(&crc_initializations);
    if (count != 1) {
        fprintf(stderr, "expected one shared CRC initialization, got %u\n", count);
        return 5;
    }
    return 0;
}
