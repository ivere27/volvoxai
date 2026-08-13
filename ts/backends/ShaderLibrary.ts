import linearF32Shader from '../../shaders/inference/linearF32.wgsl';
import linearF32TiledShader from '../../shaders/inference/linearF32Tiled.wgsl';
import linearF32RowMajorShader from '../../shaders/inference/linearF32RowMajor.wgsl';
import linearF32RowMajorTiledShader from '../../shaders/inference/linearF32RowMajorTiled.wgsl';
import linearInt8Shader from '../../shaders/inference/linearInt8.wgsl';
import linearInt8TiledShader from '../../shaders/inference/linearInt8Tiled.wgsl';
import conv2DShader from '../../shaders/inference/conv2D.wgsl';
import conv2DDepthwise8Shader from '../../shaders/inference/conv2DDepthwise8.wgsl';
import conv2DPointwise16Shader from '../../shaders/inference/conv2DPointwise16.wgsl';
import conv2DPointwise16TileShader from '../../shaders/inference/conv2DPointwise16Tile.wgsl';
import conv2DPointwise8Vec2Shader from '../../shaders/inference/conv2DPointwise8Vec2.wgsl';
import conv2DPointwise8Vec4Shader from '../../shaders/inference/conv2DPointwise8Vec4.wgsl';
import conv2DRegularC3Out16Shader from '../../shaders/inference/conv2DRegularC3Out16.wgsl';
import conv2DRegularOut16Shader from '../../shaders/inference/conv2DRegularOut16.wgsl';
import layerNormShader from '../../shaders/inference/layerNorm.wgsl';
import groupNormShader from '../../shaders/inference/groupNorm.wgsl';
import binaryBroadcastShader from '../../shaders/inference/binaryBroadcast.wgsl';
import elementwiseShader from '../../shaders/inference/elementwise.wgsl';
import resizeShader from '../../shaders/inference/resize.wgsl';
import sliceShader from '../../shaders/inference/slice.wgsl';
import sliceNdShader from '../../shaders/inference/sliceNd.wgsl';
import subShader from '../../shaders/inference/sub.wgsl';
import divShader from '../../shaders/inference/div.wgsl';
import siLUShader from '../../shaders/inference/siLU.wgsl';
import leakyReLUShader from '../../shaders/inference/leakyReLU.wgsl';
import tanhShader from '../../shaders/inference/tanh.wgsl';
import clipShader from '../../shaders/inference/clip.wgsl';
import rMSNormShader from '../../shaders/inference/rMSNorm.wgsl';
import softmaxShader from '../../shaders/inference/softmax.wgsl';
import pReLUShader from '../../shaders/inference/pReLU.wgsl';
import logSoftmaxShader from '../../shaders/inference/logSoftmax.wgsl';
import reduceShader from '../../shaders/inference/reduce.wgsl';
import argMaxF32Shader from '../../shaders/inference/argMaxF32.wgsl';
import argMaxI32Shader from '../../shaders/inference/argMaxI32.wgsl';
import argMaxI8Shader from '../../shaders/inference/argMaxI8.wgsl';
import averagePool2DShader from '../../shaders/inference/averagePool2D.wgsl';
import gatherInt32Shader from '../../shaders/inference/gatherInt32.wgsl';
import gatherElementsShader from '../../shaders/inference/gatherElements.wgsl';
import whereShader from '../../shaders/inference/where.wgsl';
import whereTypedShader from '../../shaders/inference/whereTyped.wgsl';
import castShader from '../../shaders/inference/cast.wgsl';
import dequantizeLinearTypedShader from '../../shaders/inference/dequantizeLinearTyped.wgsl';
import quantizeLinearTypedShader from '../../shaders/inference/quantizeLinearTyped.wgsl';
import qAddShader from '../../shaders/inference/qAdd.wgsl';
import qEmbeddingInt8Shader from '../../shaders/inference/qEmbeddingInt8.wgsl';
import qLinearInt8Shader from '../../shaders/inference/qLinearInt8.wgsl';
import qLinearInt8TiledShader from '../../shaders/inference/qLinearInt8Tiled.wgsl';
import qLinearInt8DotShader from '../../shaders/inference/qLinearInt8Dot.wgsl';
import qLinearInt8DotTiledShader from '../../shaders/inference/qLinearInt8DotTiled.wgsl';
import qConv2DInt8Shader from '../../shaders/inference/qConv2DInt8.wgsl';
import qConv2DInt8TiledShader from '../../shaders/inference/qConv2DInt8Tiled.wgsl';
import qConv2DInt8DotTiledShader from '../../shaders/inference/qConv2DInt8DotTiled.wgsl';
import qGELUInt8Shader from '../../shaders/inference/qGELUInt8.wgsl';
import qGroupNormApplyShader from '../../shaders/inference/qGroupNormApply.wgsl';
import qGroupNormStatsShader from '../../shaders/inference/qGroupNormStats.wgsl';
import qLayerNormApplyShader from '../../shaders/inference/qLayerNormApply.wgsl';
import qLayerNormStatsShader from '../../shaders/inference/qLayerNormStats.wgsl';
import qMaskedMeanInt8Shader from '../../shaders/inference/qMaskedMeanInt8.wgsl';
import qSDPAInt8Shader from '../../shaders/inference/qSDPAInt8.wgsl';
import qArgMaxInt8Shader from '../../shaders/inference/qArgMaxInt8.wgsl';
import qSiLUInt8Shader from '../../shaders/inference/qSiLUInt8.wgsl';
import requantizeLinearTypedShader from '../../shaders/inference/requantizeLinearTyped.wgsl';
import expandShader from '../../shaders/inference/expand.wgsl';
import expandTypedShader from '../../shaders/inference/expandTyped.wgsl';
import padShader from '../../shaders/inference/pad.wgsl';
import convTranspose2DShader from '../../shaders/inference/convTranspose2D.wgsl';
import reLUShader from '../../shaders/inference/reLU.wgsl';
import sigmoidShader from '../../shaders/inference/sigmoid.wgsl';
import hardSwishShader from '../../shaders/inference/hardSwish.wgsl';
import hardSigmoidShader from '../../shaders/inference/hardSigmoid.wgsl';
import copyShader from '../../shaders/inference/copy.wgsl';
import copyTypedShader from '../../shaders/inference/copyTyped.wgsl';
import batchNorm2DShader from '../../shaders/inference/batchNorm2D.wgsl';
import gELUShader from '../../shaders/inference/gELU.wgsl';
import addShader from '../../shaders/inference/add.wgsl';
import upsample2xShader from '../../shaders/inference/upsample2x.wgsl';
import concatCopyShader from '../../shaders/inference/concatCopy.wgsl';
import concatCopyTypedShader from '../../shaders/inference/concatCopyTyped.wgsl';
import concat2Shader from '../../shaders/inference/concat2.wgsl';
import broadcastBinaryShader from '../../shaders/inference/broadcastBinary.wgsl';
import generalTransposeShader from '../../shaders/inference/generalTranspose.wgsl';
import transposeTypedShader from '../../shaders/inference/transposeTyped.wgsl';
import splitShader from '../../shaders/inference/split.wgsl';
import profileYShader from '../../shaders/inference/profileY.wgsl';
import profileXShader from '../../shaders/inference/profileX.wgsl';
import globalAveragePoolShader from '../../shaders/inference/globalAveragePool.wgsl';
import meanHeightShader from '../../shaders/inference/meanHeight.wgsl';
import maxPool2DShader from '../../shaders/inference/maxPool2D.wgsl';
import maxPool2DTypedShader from '../../shaders/inference/maxPool2DTyped.wgsl';
import resizeNearestTypedShader from '../../shaders/inference/resizeNearestTyped.wgsl';
import interp1DShader from '../../shaders/inference/interp1D.wgsl';
import conv1DShader from '../../shaders/inference/conv1D.wgsl';
import spatialSoftargmaxYShader from '../../shaders/inference/spatialSoftargmaxY.wgsl';
import embeddingShader from '../../shaders/inference/embedding.wgsl';
import sDPAShader from '../../shaders/inference/sDPA.wgsl';
import crossSDPAShader from '../../shaders/inference/crossSDPA.wgsl';
import crossAttentionShader from '../../shaders/inference/crossAttention.wgsl';
import crossAttentionF32Shader from '../../shaders/inference/crossAttentionF32.wgsl';
import nonMaxSuppressionShader from '../../shaders/inference/nonMaxSuppression.wgsl';
import moeRouterShader from '../../shaders/inference/moeRouter.wgsl';
import moeLinearShader from '../../shaders/inference/moeLinear.wgsl';
import loraApplyShader from '../../shaders/inference/loraApply.wgsl';
import incrementalRowByteCopyShader from '../../shaders/inference/incrementalRowByteCopy.wgsl';
import batchMatMulShader from '../../shaders/inference/batchMatMul.wgsl';
import qBatchMatMulShader from '../../shaders/inference/qBatchMatMul.wgsl';
import qBatchMatMulDotShader from '../../shaders/inference/qBatchMatMulDot.wgsl';
import compareI32Shader from '../../shaders/inference/compareI32.wgsl';
import notI32Shader from '../../shaders/inference/notI32.wgsl';
import clipTypedShader from '../../shaders/inference/clipTyped.wgsl';
import concatCopy32Shader from '../../shaders/inference/concatCopy32.wgsl';
import copy32Shader from '../../shaders/inference/copy32.wgsl';

export class ShaderLibrary {
  static getLinearF32Shader() {
    return linearF32Shader;
  }
  static getLinearF32TiledShader() {
    return linearF32TiledShader;
  }
  static getLinearF32RowMajorShader() {
    return linearF32RowMajorShader;
  }
  static getLinearF32RowMajorTiledShader() {
    return linearF32RowMajorTiledShader;
  }
  static getLinearInt8Shader() {
    return linearInt8Shader;
  }
  static getLinearInt8TiledShader() {
    return linearInt8TiledShader;
  }
  static getConv2DShader() {
    return conv2DShader;
  }
  static getConv2DDepthwise8Shader() {
    return conv2DDepthwise8Shader;
  }
  static getConv2DPointwise16Shader() {
    return conv2DPointwise16Shader;
  }
  static getConv2DPointwise16TileShader() {
    return conv2DPointwise16TileShader;
  }
  static getConv2DPointwise8Vec2Shader() {
    return conv2DPointwise8Vec2Shader;
  }
  static getConv2DPointwise8Vec4Shader() {
    return conv2DPointwise8Vec4Shader;
  }
  static getConv2DRegularC3Out16Shader() {
    return conv2DRegularC3Out16Shader;
  }
  static getConv2DRegularOut16Shader() {
    return conv2DRegularOut16Shader;
  }
  static getLayerNormShader() {
    return layerNormShader;
  }
  static getGroupNormShader() {
    return groupNormShader;
  }
  static getBinaryBroadcastShader() {
    return binaryBroadcastShader;
  }
  static getElementwiseShader() {
    return elementwiseShader;
  }
  static getResizeShader() {
    return resizeShader;
  }
  static getSliceShader() {
    return sliceShader;
  }
  static getSliceNdShader() {
    return sliceNdShader;
  }
  static getSubShader() {
    return subShader;
  }
  static getDivShader() {
    return divShader;
  }
  static getSiLUShader() {
    return siLUShader;
  }
  static getLeakyReLUShader() {
    return leakyReLUShader;
  }
  static getTanhShader() {
    return tanhShader;
  }
  static getClipShader() {
    return clipShader;
  }
  static getRMSNormShader() {
    return rMSNormShader;
  }
  static getSoftmaxShader() {
    return softmaxShader;
  }
  static getPReLUShader() {
    return pReLUShader;
  }
  static getLogSoftmaxShader() {
    return logSoftmaxShader;
  }
  static getReduceShader() {
    return reduceShader;
  }
  static getArgMaxF32Shader() {
    return argMaxF32Shader;
  }
  static getArgMaxI32Shader() {
    return argMaxI32Shader;
  }
  static getArgMaxI8Shader() {
    return argMaxI8Shader;
  }
  static getAveragePool2DShader() {
    return averagePool2DShader;
  }
  static getGatherInt32Shader() {
    return gatherInt32Shader;
  }
  static getGatherElementsShader() {
    return gatherElementsShader;
  }
  static getWhereShader() {
    return whereShader;
  }
  static getWhereTypedShader() {
    return whereTypedShader;
  }
  static getCastShader() {
    return castShader;
  }
  static getDequantizeLinearShader() {
    return dequantizeLinearTypedShader;
  }
  static getQuantizeLinearShader() {
    return quantizeLinearTypedShader;
  }
  static getQAddShader() {
    return qAddShader;
  }
  static getQEmbeddingShader() {
    return qEmbeddingInt8Shader;
  }
  static getQLinearShader() {
    return qLinearInt8Shader;
  }
  static getQLinearTiledShader() {
    return qLinearInt8TiledShader;
  }
  static getQLinearDotShader() {
    return qLinearInt8DotShader;
  }
  static getQLinearDotTiledShader() {
    return qLinearInt8DotTiledShader;
  }
  static getQConv2DShader() {
    return qConv2DInt8Shader;
  }
  static getQConv2DTiledShader() {
    return qConv2DInt8TiledShader;
  }
  static getQConv2DDotTiledShader() {
    return qConv2DInt8DotTiledShader;
  }
  static getQGELUShader() {
    return qGELUInt8Shader;
  }
  static getQGroupNormApplyShader() {
    return qGroupNormApplyShader;
  }
  static getQGroupNormStatsShader() {
    return qGroupNormStatsShader;
  }
  static getQLayerNormApplyShader() {
    return qLayerNormApplyShader;
  }
  static getQLayerNormStatsShader() {
    return qLayerNormStatsShader;
  }
  static getQMaskedMeanShader() {
    return qMaskedMeanInt8Shader;
  }
  static getQSDPAShader() {
    return qSDPAInt8Shader;
  }
  static getQArgMaxShader() {
    return qArgMaxInt8Shader;
  }
  static getQSiLUShader() {
    return qSiLUInt8Shader;
  }
  static getRequantizeLinearShader() {
    return requantizeLinearTypedShader;
  }
  static getExpandShader() {
    return expandShader;
  }
  static getTypedExpandShader() {
    return expandTypedShader;
  }
  static getPadShader() {
    return padShader;
  }
  static getConvTranspose2DShader() {
    return convTranspose2DShader;
  }
  static getReLUShader() {
    return reLUShader;
  }
  static getSigmoidShader() {
    return sigmoidShader;
  }
  static getHardSwishShader() {
    return hardSwishShader;
  }
  static getHardSigmoidShader() {
    return hardSigmoidShader;
  }
  static getCopyShader() {
    return copyShader;
  }
  static getTypedCopyShader() {
    return copyTypedShader;
  }
  static getBatchNorm2DShader() {
    return batchNorm2DShader;
  }
  static getGELUShader() {
    return gELUShader;
  }
  static getAddShader() {
    return addShader;
  }
  static getUpsample2xShader() {
    return upsample2xShader;
  }
  static getConcatCopyShader() {
    return concatCopyShader;
  }
  static getTypedConcatCopyShader() {
    return concatCopyTypedShader;
  }
  static getConcat2Shader() {
    return concat2Shader;
  }
  static getBroadcastBinaryShader(binOp = "out_val = av + bv;") {
    // Substitute the per-op expression (Add/Mul/Sub/Div) into the shared template.
    return broadcastBinaryShader.replace("//__BINOP__", binOp);
  }
  static getBroadcastAddShader() {
    return this.getBroadcastBinaryShader("out_val = av + bv;");
  }
  static getBroadcastAddReLUShader() {
    return this.getBroadcastBinaryShader(
      "out_val = av + bv; if (out_val < 0.0) { out_val = 0.0; }",
    );
  }
  static getBroadcastAddReLU6Shader() {
    return this.getBroadcastBinaryShader(
      "out_val = av + bv; if (out_val < 0.0) { out_val = 0.0; } if (out_val > 6.0) { out_val = 6.0; }",
    );
  }
  static getBroadcastMulShader() {
    return this.getBroadcastBinaryShader("out_val = av * bv;");
  }
  static getBroadcastSubShader() {
    return this.getBroadcastBinaryShader("out_val = av - bv;");
  }
  static getBroadcastDivShader() {
    return this.getBroadcastBinaryShader("out_val = av / bv;");
  }
  static getGeneralTransposeShader() {
    return generalTransposeShader;
  }
  static getTypedTransposeShader() {
    return transposeTypedShader;
  }
  static getSplitShader() {
    return splitShader;
  }
  static getProfileYShader() {
    return profileYShader;
  }
  static getProfileXShader() {
    return profileXShader;
  }
  static getGlobalAveragePoolShader() {
    return globalAveragePoolShader;
  }
  static getMeanHeightShader() {
    return meanHeightShader;
  }
  static getMaxPool2DShader() {
    return maxPool2DShader;
  }
  static getTypedMaxPool2DShader() {
    return maxPool2DTypedShader;
  }
  static getTypedResizeNearestShader() {
    return resizeNearestTypedShader;
  }
  static getInterp1DShader() {
    return interp1DShader;
  }
  static getConv1DShader() {
    return conv1DShader;
  }
  static getSpatialSoftargmaxYShader() {
    return spatialSoftargmaxYShader;
  }
  static getEmbeddingShader() {
    return embeddingShader;
  }
  static getSDPAShader() {
    return sDPAShader;
  }
  static getCrossSDPAShader() {
    return crossSDPAShader;
  }
  static getCrossAttentionShader() {
    return crossAttentionShader;
  }
  static getCrossAttentionF32Shader() {
    return crossAttentionF32Shader;
  }
  static getNonMaxSuppressionShader() {
    return nonMaxSuppressionShader;
  }
  static getMoERouterShader() {
    return moeRouterShader;
  }
  static getMoELinearShader() {
    return moeLinearShader;
  }
  static getLoRAApplyShader() {
    return loraApplyShader;
  }
  static getIncrementalRowByteCopyShader() {
    return incrementalRowByteCopyShader;
  }
  static getBatchMatMulShader() {
    return batchMatMulShader;
  }
  static getQBatchMatMulShader() {
    return qBatchMatMulShader;
  }
  static getQBatchMatMulDotShader() {
    return qBatchMatMulDotShader;
  }
  static getCompareI32Shader() {
    return compareI32Shader;
  }
  static getNotI32Shader() {
    return notI32Shader;
  }
  static getTypedClipShader() {
    return clipTypedShader;
  }
  static getConcatCopy32Shader() {
    return concatCopy32Shader;
  }
  static getCopy32Shader() {
    return copy32Shader;
  }
}
