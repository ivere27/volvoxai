export class GraphOperatorNormalizer {
  /**
   * MatMul weights may be output-major [d_out,d_in] (out=x·Wᵀ) or input-major
   * [d_in,d_out] (out=x·W). Non-square weights disambiguate by shape; square ones
   * cannot, so infer the graph-wide convention from unambiguous weights and tag
   * every MatMul node with `wLayout` ('dout' | 'din').
   */
  static resolveMatMulLayouts(graph) {
    const isMM = (n) => n.opType === "MatMul" || n.opType === "Linear" || n.opType === "Gemm";
    const quantScale = (n) => n.inputs.scale || n.inputs.weight_scale;
    const explicitLayout = (n) => {
      const layout = n.params?.weight_layout;
      if (layout == null || layout === "") return n.params?.transB ? "dout" : null;
      if (["OUT_IN", "out_in", "OI", "peft"].includes(layout)) return "dout";
      if (["IN_OUT", "in_out", "IO", "din_dout"].includes(layout)) return "din";
      throw new Error(`[GraphLoader] Unsupported linear weight_layout '${layout}' at node ${n.id}.`);
    };
    const dims = (n) => {
      const K = n.inputs.input?.shape?.[n.inputs.input.shape.length - 1];
      const outT: any = Object.values(n.outputs)[0];
      const N = outT?.shape?.[outT.shape.length - 1];
      return [K, N];
    };
    let din = 0, dout = 0;
    for (const n of graph.nodes) {
      if (!isMM(n) || !n.inputs.weight || quantScale(n)) continue;
      const explicit = explicitLayout(n);
      if (explicit === "din") { din++; continue; }
      if (explicit === "dout") { dout++; continue; }
      const w = n.inputs.weight.shape; if (!w || w.length < 2) continue;
      const [K, N] = dims(n); if (K === N) continue;
      if (w[0] === N && w[1] === K) dout++;
      else if (w[0] === K && w[1] === N) din++;
    }
    const model = din > dout ? "din" : "dout";
    for (const n of graph.nodes) {
      if (!isMM(n)) continue;
      const w = n.inputs.weight?.shape; const [K, N] = dims(n);
      if (quantScale(n)) n.wLayout = "dout";                           // Quantized linear is [d_out, d_in]
      else if (explicitLayout(n)) n.wLayout = explicitLayout(n);
      else if (w && w.length >= 2 && K !== N && w[0] === N && w[1] === K) n.wLayout = "dout";
      else if (w && w.length >= 2 && K !== N && w[0] === K && w[1] === N) n.wLayout = "din";
      else n.wLayout = model;                                          // square → model convention
    }
  }


  /**
   * Ordinary Conv kernels require float32 weights. QConv2D is deliberately
   * rejected above rather than silently converted, because its activation
   * quantization semantics cannot be preserved by a weight-only conversion.
   */
  static dequantizeConvWeights(graph) {
    for (const node of graph.nodes) {
      if (node.opType !== "Conv2D" && node.opType !== "Conv1D") continue;
      const w = node.inputs.weight;
      const s = node.inputs.scale || node.inputs.weight_scale;
      if (!s || !w || w.dtype !== "int8" || !w.buffer) continue;
      const zp = node.inputs.weight_zero_point;
      const outC = w.shape[0];
      const perOut = w.buffer.length / outC;
      const deq = new Float32Array(w.buffer.length);
      for (let oc = 0; oc < outC; oc++) {
        const sc = s.buffer[oc];
        const z = zp && zp.buffer ? zp.buffer[zp.buffer.length === 1 ? 0 : oc] : 0;
        const base = oc * perOut;
        for (let j = 0; j < perOut; j++) deq[base + j] = (w.buffer[base + j] - z) * sc;
      }
      w.buffer = deq;
      w.dtype = "float32";
      w.quantization = null;
      w.sizeBytes = deq.length * 4;
      delete node.inputs.scale;
      delete node.inputs.weight_scale;
      delete node.inputs.weight_zero_point;
    }
  }


  static normalizeConvWeightsForImageLayout(graph) {
    const normalizedWeights = new Map();
    for (const node of graph.nodes) {
      if (node.opType !== "Conv2D") continue;
      const w = node.inputs.weight;
      const input = node.inputs.input;
      const output = node.outputs.out;
      if (!w || !w.buffer || !input || !output || w.shape.length !== 4) continue;
      const layout = node.params?.weight_layout || "HWIO";
      const prior = normalizedWeights.get(w);
      if (prior) {
        if (layout !== prior.source && layout !== prior.canonical) {
          throw new Error(
            `[GraphLoader] Shared Conv2D weight '${w.name}' has conflicting layouts ` +
            `'${prior.source}' and '${layout}'.`,
          );
        }
        node.params.weight_layout = prior.canonical;
        continue;
      }
      if (layout === "HWIO" || layout === "HWCM") {
        normalizedWeights.set(w, { source: layout, canonical: layout });
        continue;
      }

      if (layout === "OHWI" || layout === "OIHW") {
        const [ocN, a, b, c] = w.shape;
        const kh = layout === "OHWI" ? a : b;
        const kw = layout === "OHWI" ? b : c;
        const icN = layout === "OHWI" ? c : a;
        const src = w.buffer;
        const dst = new Float32Array(src.length);
        for (let oc = 0; oc < ocN; oc++) {
          for (let y = 0; y < kh; y++) {
            for (let x = 0; x < kw; x++) {
              for (let ic = 0; ic < icN; ic++) {
                dst[(((y * kw + x) * icN + ic) * ocN) + oc] =
                  layout === "OHWI"
                    ? src[(((oc * kh + y) * kw + x) * icN) + ic]
                    : src[(((oc * icN + ic) * kh + y) * kw) + x];
              }
            }
          }
        }
        w.buffer = dst;
        w.shape = [kh, kw, icN, ocN];
        w.sizeBytes = dst.length * 4;
        node.params.weight_layout = "HWIO";
        normalizedWeights.set(w, { source: layout, canonical: "HWIO" });
      } else if (layout === "1HWO" || layout === "1HWM") {
        const [, kh, kw, ocN] = w.shape;
        const icN = input.shape[3];
        const mult = output.shape[3] / icN;
        if (!Number.isInteger(mult) || mult <= 0 || icN * mult !== ocN) {
          throw new Error(`[GraphLoader] Invalid depthwise Conv2D shape for node ${node.id}`);
        }
        const src = w.buffer;
        const dst = new Float32Array(src.length);
        for (let y = 0; y < kh; y++) {
          for (let x = 0; x < kw; x++) {
            for (let ic = 0; ic < icN; ic++) {
              for (let m = 0; m < mult; m++) {
                dst[(((y * kw + x) * icN + ic) * mult) + m] =
                  src[((y * kw + x) * ocN) + ic * mult + m];
              }
            }
          }
        }
        w.buffer = dst;
        w.shape = [kh, kw, icN, mult];
        w.sizeBytes = dst.length * 4;
        node.params.weight_layout = "HWCM";
        normalizedWeights.set(w, { source: layout, canonical: "HWCM" });
      } else {
        throw new Error(`[GraphLoader] Unsupported Conv2D weight_layout '${layout}'. VolvoxAI uses NHWC/HWIO only.`);
      }
    }
  }
}
