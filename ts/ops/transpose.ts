export function _cpuTranspose(node) {

    const input = node.inputs.input;
    const inBuf = input.buffer;
    const inShape = input.shape;
    const outBuf = node.outputs.out.buffer;
    const perm = node.params.perm || [...Array(inShape.length).keys()].reverse();
    
    const inStrides = new Array(inShape.length);
    let s = 1;
    for (let i = inShape.length - 1; i >= 0; i--) {
        inStrides[i] = s;
        s *= inShape[i];
    }
    const outShape = perm.map(p => inShape[p]);
    const outStrides = new Array(outShape.length);
    s = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
        outStrides[i] = s;
        s *= outShape[i];
    }
    
    const elements = inBuf.length;
    for (let i = 0; i < elements; i++) {
        let inIdx = 0;
        let temp = i;
        for (let j = 0; j < outShape.length; j++) {
            const outCoord = Math.floor(temp / outStrides[j]);
            temp %= outStrides[j];
            inIdx += outCoord * inStrides[perm[j]];
        }
        outBuf[i] = inBuf[inIdx];
    }
  }