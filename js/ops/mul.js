export function _cpuMul(node) {

    const a = node.inputs.a;
    const b = node.inputs.b;
    const outBuf = node.outputs.out.buffer;
    let aBuf = a.buffer;
    let bBuf = b.buffer;
    let aShape = a.shape;
    let bShape = b.shape;
    
    // Swap to ensure a is the larger tensor
    if (aBuf.length < bBuf.length) {
        let temp = aBuf; aBuf = bBuf; bBuf = temp;
        let tempS = aShape; aShape = bShape; bShape = tempS;
    }
    
    const elements = aBuf.length;
    if (bBuf.length === 1) {
      for (let i = 0; i < elements; i++) outBuf[i] = aBuf[i] * bBuf[0];
    } else if (bBuf.length === elements) {
      for (let i = 0; i < elements; i++) outBuf[i] = aBuf[i] * bBuf[i];
    } else if (aShape.length === 4 && bBuf.length === aShape[3]) {
      const c = aShape[3];
      for (let i = 0; i < elements; i++) outBuf[i] = aBuf[i] * bBuf[i % c];
    } else if (aShape.length === 4 && bShape.length === 4 && bShape[0] === 1 && bShape[1] === 1 && bShape[2] === 1 && bShape[3] === aShape[3]) {
      const c = aShape[3];
      for (let i = 0; i < elements; i++) outBuf[i] = aBuf[i] * bBuf[i % c];
    } else {
      console.warn("[VolvoxAI CPU] Executing Mul is not fully implemented yet for shapes", a.shape, b.shape);
    }
  }
