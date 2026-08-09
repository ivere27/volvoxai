import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import {
  assertCanonicalLayout,
  assertDistinctOutputStorage,
  checkedTransposeOutput,
  spatialKernelPorts,
  spatialPair,
} from './spatialKernelValidation.js';

/**
 * ConvTranspose2D over NHWC activations and HWIO weights [kh, kw, in_c, out_c].
 *
 * The kernel gathers rather than scatters: every output pixel is computed once
 * from the input taps that reach it, instead of accumulating into shared output
 * pixels from each input. That removes the read-modify-write traffic and the
 * zero-fill pass the scatter form needed, and it is what lets the innermost
 * loop run contiguously over out_c in both the weight and the output row --
 * the same reason ordinary Conv2D indexes HWIO.
 */
export function _cpuConvTranspose2D(node) {
    const operation = 'ConvTranspose2D';
    const ports = spatialKernelPorts(
        node,
        [['input', 'x'], ['weight']],
        ['bias'],
        operation,
    );
    const [input, weight] = ports.inputs;
    const bias = ports.optional.bias;
    const output = ports.output;
    assertShapeKernelTensor(input, `${operation} input`, {
        dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
    });
    assertShapeKernelTensor(weight, `${operation} weight`, {
        dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
    });
    const params = assertShapeKernelParams(
        node,
        ['kernel', 'stride', 'padding', 'data_layout', 'weight_layout'],
        operation,
    );
    assertCanonicalLayout(params.data_layout, 'NHWC', operation, 'data_layout');
    assertCanonicalLayout(params.weight_layout, 'HWIO', operation, 'weight_layout');
    const [kernelHeight, kernelWidth] = spatialPair(
        params.kernel,
        1,
        operation,
        'kernel',
        false,
        true,
    );
    const [strideHeight, strideWidth] = spatialPair(
        params.stride,
        1,
        operation,
        'stride',
        false,
    );
    const [paddingHeight, paddingWidth] = spatialPair(
        params.padding,
        0,
        operation,
        'padding',
        true,
    );

    const inBuf = input.buffer;
    const wBuf = weight.buffer;
    const bBuf = bias ? bias.buffer : null;
    const outBuf = output.buffer;

    const [batch, inputHeight, inputWidth, inputChannels] = input.shape;
    if (weight.shape[0] !== kernelHeight || weight.shape[1] !== kernelWidth) {
        throw new Error(`${operation} kernel must equal the HWIO weight kernel extents.`);
    }
    if (weight.shape[2] !== inputChannels) {
        throw new Error(`${operation} weight input channels must match the activation.`);
    }
    const outputChannels = weight.shape[3];
    if (bias !== undefined) {
        assertShapeKernelTensor(bias, `${operation} bias`, {
            dtypes: ['float32'], minimumRank: 1, maximumRank: 1,
        });
        if (bias.shape[0] !== outputChannels) {
            throw new Error(`${operation} bias shape must be [${outputChannels}].`);
        }
    }
    const outputHeight = checkedTransposeOutput(
        inputHeight,
        kernelHeight,
        strideHeight,
        paddingHeight,
        `${operation} output height`,
    );
    const outputWidth = checkedTransposeOutput(
        inputWidth,
        kernelWidth,
        strideWidth,
        paddingWidth,
        `${operation} output width`,
    );
    assertShapeKernelOutput(
        output,
        [batch, outputHeight, outputWidth, outputChannels],
        'float32',
        undefined,
        operation,
    );
    assertDistinctOutputStorage(output, [input, weight, bias], operation);

    for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
        for (let outputY = 0; outputY < outputHeight; outputY++) {
            for (let outputX = 0; outputX < outputWidth; outputX++) {
                const outBase =
                    ((batchIndex * outputHeight + outputY) * outputWidth + outputX) *
                    outputChannels;
                for (let oc = 0; oc < outputChannels; oc++) {
                    outBuf[outBase + oc] = bBuf ? bBuf[oc] : 0.0;
                }
                for (let kernelY = 0; kernelY < kernelHeight; kernelY++) {
                    const shiftedY = outputY + paddingHeight - kernelY;
                    if (shiftedY < 0 || shiftedY % strideHeight !== 0) continue;
                    const inputY = shiftedY / strideHeight;
                    if (inputY >= inputHeight) continue;
                    for (let kernelX = 0; kernelX < kernelWidth; kernelX++) {
                        const shiftedX = outputX + paddingWidth - kernelX;
                        if (shiftedX < 0 || shiftedX % strideWidth !== 0) continue;
                        const inputX = shiftedX / strideWidth;
                        if (inputX >= inputWidth) continue;
                        const inBase =
                            ((batchIndex * inputHeight + inputY) * inputWidth + inputX) *
                            inputChannels;
                        const wBase =
                            (kernelY * kernelWidth + kernelX) * inputChannels * outputChannels;
                        for (let ic = 0; ic < inputChannels; ic++) {
                            const value = inBuf[inBase + ic];
                            const wRow = wBase + ic * outputChannels;
                            for (let oc = 0; oc < outputChannels; oc++) {
                                outBuf[outBase + oc] += value * wBuf[wRow + oc];
                            }
                        }
                    }
                }
            }
        }
    }
}
