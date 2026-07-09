#ifndef ENGINE_H
#define ENGINE_H

// Native forward-pass engine: parses a Volvox blueprint (config.json + safetensors),
// builds the graph once, and runs it node-by-node on the CPU/GPU kernels — the C
// counterpart of the JS WasmEngine/CPUEngine dispatch loop.
//
// Two ways to use it:
//   * Single pass:  engine_run(config, weights, input_file, output_file)
//   * Autoregressive: engine_init() once, then per token: poke engine_input_ptr(),
//     engine_forward(), read engine_last_logits(). engine_free_ctx() when done.
//     This loads the weights + builds the graph ONCE (no per-token reload/leak).

int    engine_init(const char* config_path, const char* weights_path);
float* engine_input_ptr(const char* name, long* numel);   // in-memory input tensor
float* engine_tensor_ptr(const char* name, long* numel);   // any graph tensor by name
int    engine_tensor_info(const char* name, long* numel, int* shape, int* ndim);
int    engine_forward(void);                              // run the whole graph (all rows)
int    engine_prefill(int n_tokens);                     // process the prompt, fill K/V caches
int    engine_decode(int pos);                           // process one new position via the cache
void   engine_profile_reset(void);                       // reset debug per-op timing aggregation
void   engine_profile_report(void);                      // print debug per-op timing aggregation
const float* engine_last_logits(int* count);             // last-token row (uses g_last_token)
void   engine_free_ctx(void);

// Convenience single-pass wrapper (loads input file, runs once, writes output file).
int engine_run(const char* config_path, const char* weights_path,
               const char* input_path, const char* output_path);

#endif
