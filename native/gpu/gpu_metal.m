#include <stdio.h>
#include <stdlib.h>
// #import <Metal/Metal.h>

typedef struct {
    // id<MTLDevice> device;
    // id<MTLCommandQueue> commandQueue;
} MetalContext;

MetalContext* metal_init() {
    printf("[Metal] Initializing Metal Compute Context...\n");
    MetalContext* ctx = malloc(sizeof(MetalContext));
    // ctx->device = MTLCreateSystemDefaultDevice();
    // ctx->commandQueue = [ctx->device newCommandQueue];
    return ctx;
}

void metal_load_shader(MetalContext* ctx, const char* metal_filepath) {
    printf("[Metal] Loading MSL shader from: %s\n", metal_filepath);
    // TODO: Read .metal text file
    // TODO: Compile to MTLLibrary
}

void metal_run_compute(MetalContext* ctx, void* input_data, size_t input_size) {
    printf("[Metal] Running compute pipeline...\n");
    // TODO: Allocate MTLBuffer, encode compute command, commit
}

void metal_cleanup(MetalContext* ctx) {
    printf("[Metal] Cleaning up context...\n");
    free(ctx);
}
