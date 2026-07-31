# 30 Core Operator Fusion Patterns for High-Performance Inference Engines

Status: this page is a design catalogue, not VolvoxAI optimizer or backend
coverage. A pattern is implemented only when it appears in the verified typed
pass inventory and has matching legality, differential-correctness, target
route, and kernel tests. See [graph-optimizer-design.md](graph-optimizer-design.md)
and [operation_list.md](operation_list.md) for current behavior.

## Introduction
In edge AI and highly optimized inference engines (like XNNPACK, TFLite, and VolvoxAI), memory bandwidth is the primary bottleneck, not CPU compute limits. Operator Fusion addresses this "Memory Wall" by combining multiple computational nodes into a single microkernel. This keeps intermediate activations resident in L1 cache or CPU registers, drastically reducing slow DRAM read/write round-trips.

However, implementing every possible fusion combination leads to "combinatorial explosion" (massive binary bloat). Therefore, lightweight engines focus on a targeted subset of high-value fusions and utilize **selective builds** to compile only what the target model needs.

Here are the 30 core fusion patterns that cover the vast majority of modern AI architectures.

---

## 1. Convolution Family (Core of Vision/CNNs)
These patterns account for 80%+ of the execution time in Object Detection and Image Classification models.

1. **Conv2D + Bias:** The fundamental block.
2. **Conv2D + Bias + ReLU / ReLU6:** The most ubiquitous standard block.
3. **Conv2D + Bias + HardSwish / Mish:** Heavily used in modern YOLO architectures.
4. **Conv2D + Add:** The core of ResNet residual connections. (In-register residual addition).
5. **Conv2D + Bias + Add + ReLU:** The complete end-stage of a ResNet block.
6. **Depthwise Conv2D + Bias + Activation:** Essential for MobileNet efficiency.
7. **Depthwise Conv2D + Pointwise Conv2D (Separable Conv):** Fused directly to avoid writing the intermediate depthwise output to memory.
8. **Transposed Conv2D (Deconv) + Bias + Activation:** Used in segmentation and generative models.
9. **Conv1D + Bias + Activation:** Standard for audio and time-series analysis.
10. **Conv3D + Bias + Activation:** Standard for video processing.

## 2. MatMul / Dense Family (Core of Transformers/LLMs)
Critical for avoiding massive memory bottlenecks in Large Language Models and Attention mechanisms.

11. **MatMul + Bias (Fully Connected/Linear):** Standard linear transformation.
12. **MatMul + Bias + GELU / SiLU:** The core of Transformer MLP blocks.
13. **MatMul + Add:** Residual connections within Transformers.
14. **MatMul + Softmax:** The fundamental backbone of the Attention mechanism.
15. **BatchMatMul + Scale + Mask + Softmax:** The "FlashAttention" pattern, computing the entire attention block in-register.
16. **MatMul + Sigmoid + Mul (GLU/SwiGLU):** Essential for modern LLMs like LLaMA.

## 3. Element-wise Family (Math & Activations)
Though computationally light, these must be fused to prevent catastrophic memory round-trips.

17. **Add + ReLU:** Post-residual activation.
18. **Mul + Add (FMA):** Fused Multiply-Add (often hardware-accelerated via single SIMD instruction).
19. **Mul + Sigmoid:** Equivalent to the SiLU activation (`x * sigmoid(x)`).
20. **Add + Add / Mul + Mul:** Consecutive identical operations folded together.
21. **Add + Clamp (Min/Max):** Value clipping/bounding (e.g., ReLU6 is just `Clamp(0, 6)`).
22. **Exp + Sum + Div:** Granular operations of a Softmax fused back together.

## 4. Normalization Family
Norm layers scan the entire memory tensor, making them extremely slow unless fused with adjacent ops.

23. **LayerNorm + MatMul:** Common in Transformers, performing linear transform immediately after normalization.
24. **RMSNorm + Mul:** A lighter normalization fusion used heavily in LLaMA.
25. **InstanceNorm + Activation:** Common in Style Transfer and generative networks.

## 5. Miscellaneous (Pooling, Quantization & Special Ops)
26. **GlobalAveragePooling + Flatten:** Squashes spatial dimensions into a 1D vector in a single pass.
27. **MaxPool + Activation:** Standard downsampling block.
28. **ArgMax + Gather:** Finds the highest probability index and retrieves the corresponding data/token.
29. **Split (Chunk) + MatMul:** Splitting Q, K, V attention vectors and immediately computing.
30. **Quantize + Conv/Add + Dequantize (Requantization):** Critical for INT8 pipelines. Computes integer math and scales back to floating-point without ever storing intermediate Int32 sums to DRAM.
