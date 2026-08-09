import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import { checkedShapeMultiply } from './shapeSystem.js';

export type VisionProfileKernelKind =
  'mean-height' | 'softargmax-y' | 'profile-x' | 'profile-y';

export function visionProfileKernelDescriptor(
  node: any,
  operation: string,
  kind: VisionProfileKernelKind,
) {
  const input = node.inputs?.input;
  const output = node.outputs?.out;
  assertShapeKernelParams(node, [], operation);
  assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
  });
  const [n, h, w, c] = input.shape;
  const expectedShape = kind === 'profile-x'
    ? [n, checkedShapeMultiply(c, 2, `${operation} channel extent`), w]
    : kind === 'profile-y'
      ? [n, checkedShapeMultiply(c, 2, `${operation} channel extent`), h]
      : [n, c, w];
  assertShapeKernelOutput(
    output, expectedShape, 'float32', undefined, operation,
  );
  return Object.freeze({ input, output, n, h, w, c });
}
