export function _cpuGather(node) {

    const input = node.inputs.input;
    const indices = node.inputs.indices;
    const out = node.outputs.out;
    let axis = node.params.axis || 0;
    if (axis < 0) axis += input.shape.length;
    
    const inShape = input.shape;
    const outShape = out.shape;
    const idxShape = indices.shape;
    
    let inStrides = new Array(inShape.length);
    let s = 1; for (let i = inShape.length - 1; i >= 0; i--) { inStrides[i] = s; s *= inShape[i]; }
    
    let outStrides = new Array(outShape.length);
    s = 1; for (let i = outShape.length - 1; i >= 0; i--) { outStrides[i] = s; s *= outShape[i]; }
    
    let idxStrides = new Array(idxShape.length);
    s = 1; for (let i = idxShape.length - 1; i >= 0; i--) { idxStrides[i] = s; s *= idxShape[i]; }
    
    for (let i = 0; i < out.buffer.length; i++) {
        let temp = i;
        let outCoords = new Array(outShape.length);
        for (let d = 0; d < outShape.length; d++) {
            outCoords[d] = Math.floor(temp / outStrides[d]);
            temp %= outStrides[d];
        }
        
        let idxOffset = 0;
        for (let d = 0; d < idxShape.length; d++) {
            idxOffset += outCoords[axis + d] * idxStrides[d];
        }
        
        let gatherIdx = indices.buffer[idxOffset];
        if (gatherIdx < 0) {
            out.buffer[i] = -1;
            continue;
        }
        let inOffset = 0;
        for (let d = 0; d < axis; d++) {
            inOffset += outCoords[d] * inStrides[d];
        }
        inOffset += gatherIdx * inStrides[axis];
        for (let d = axis + 1; d < inShape.length; d++) {
            inOffset += outCoords[d - 1 + idxShape.length] * inStrides[d];
        }
        
        out.buffer[i] = input.buffer[inOffset];
    }
  }