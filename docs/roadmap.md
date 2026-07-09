# Roadmap

Current known gaps:

- WebGPU multi-output readback should return a map like WASM/CPU.
- WebNN needs additional one-to-one builder mappings and composite formulas:
  `RMSNorm`, `CrossSDPA`, `LogSoftmax`, `DequantizeLinear`, and `Conv1D`.
- Native/WebGPU shader coverage is still missing or limited for `GatherElements`,
  `ArgMax`, `NonMaxSuppression`, and general-axis GPU `Gather`.
- Softmax and LogSoftmax shaders currently focus on the last axis.
- Browser text generation needs a higher-level streaming/autoregressive helper.
- INT4 weight formats are not implemented.
- Tokenizer parity tests against GPT-2/Neo reference BPE should be committed.
- A cross-tier whole-model parity and benchmark harness should be added to CI.

Native note: optimized Linux builds in headless VMs can expose platform Vulkan loader
issues, especially with Mesa/lavapipe. Treat real GPU validation separately from
software-loader smoke tests.

