export function _cpuSlice(node) {
    const input = node.inputs.input || node.inputs.data;
    const outBuf = node.outputs.out.buffer;
    const starts = node.params.starts || [0, 0, 0, 0];
    const steps = node.params.steps || [1, 1, 1, 1];
    const axes = node.params.axes || [0, 1, 2, 3];

    const in_s = [1, 1, 1, 1].slice(0, 4 - input.shape.length).concat(input.shape);
    const out_s = [1, 1, 1, 1].slice(0, 4 - node.outputs.out.shape.length).concat(node.outputs.out.shape);

    const st = [0, 0, 0, 0];
    const sp = [1, 1, 1, 1];
    for (let i = 0; i < axes.length; i++) {
        let ax = axes[i];
        if (ax < 0) ax += input.shape.length;
        ax += (4 - input.shape.length);
        st[ax] = starts[i] < 0 ? starts[i] + in_s[ax] : starts[i];
        sp[ax] = steps[i];
    }

    let outIdx = 0;
    for (let i0 = 0; i0 < out_s[0]; i0++) {
        for (let i1 = 0; i1 < out_s[1]; i1++) {
            for (let i2 = 0; i2 < out_s[2]; i2++) {
                for (let i3 = 0; i3 < out_s[3]; i3++) {
                    const src0 = st[0] + i0 * sp[0];
                    const src1 = st[1] + i1 * sp[1];
                    const src2 = st[2] + i2 * sp[2];
                    const src3 = st[3] + i3 * sp[3];
                    const inIdx = src0 * (in_s[1] * in_s[2] * in_s[3]) + src1 * (in_s[2] * in_s[3]) + src2 * in_s[3] + src3;
                    outBuf[outIdx++] = input.buffer[inIdx];
                }
            }
        }
    }
}
