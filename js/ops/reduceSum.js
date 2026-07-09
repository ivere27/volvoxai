export function _cpuReduceSum(node) {
    const input = node.inputs.input || node.inputs.data;
    const inBuf = input.buffer;
    const outBuf = node.outputs.out.buffer;
    
    // Simplistic reduce: flatten outer dimensions as batch, and reduce inner dimension
    // Typically used for [B, D] -> [B, 1] or similar
    const in_shape = input.shape.length === 2 ? input.shape : [1, input.buffer.length];
    const b = in_shape[0];
    const d = in_shape[1];
    
    for (let i = 0; i < b; i++) {
        let sum = 0.0;
        for (let j = 0; j < d; j++) {
            sum += inBuf[i * d + j];
        }
        outBuf[i] = sum;
    }
}
