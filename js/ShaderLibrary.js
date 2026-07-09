import linearF32Shader from '../shaders/linearF32.wgsl';
import linearInt8Shader from '../shaders/linearInt8.wgsl';
import conv2DShader from '../shaders/conv2D.wgsl';
import conv2DDepthwise8Shader from '../shaders/conv2DDepthwise8.wgsl';
import conv2DPointwise16Shader from '../shaders/conv2DPointwise16.wgsl';
import conv2DPointwise16TileShader from '../shaders/conv2DPointwise16Tile.wgsl';
import conv2DPointwise8Vec2Shader from '../shaders/conv2DPointwise8Vec2.wgsl';
import conv2DPointwise8Vec4Shader from '../shaders/conv2DPointwise8Vec4.wgsl';
import conv2DRegularC3Out16Shader from '../shaders/conv2DRegularC3Out16.wgsl';
import layerNormShader from '../shaders/layerNorm.wgsl';
import binaryBroadcastShader from '../shaders/binaryBroadcast.wgsl';
import elementwiseShader from '../shaders/elementwise.wgsl';
import resizeShader from '../shaders/resize.wgsl';
import sliceShader from '../shaders/slice.wgsl';
import subShader from '../shaders/sub.wgsl';
import divShader from '../shaders/div.wgsl';
import siLUShader from '../shaders/siLU.wgsl';
import leakyReLUShader from '../shaders/leakyReLU.wgsl';
import tanhShader from '../shaders/tanh.wgsl';
import clipShader from '../shaders/clip.wgsl';
import rMSNormShader from '../shaders/rMSNorm.wgsl';
import softmaxShader from '../shaders/softmax.wgsl';
import pReLUShader from '../shaders/pReLU.wgsl';
import logSoftmaxShader from '../shaders/logSoftmax.wgsl';
import reduceShader from '../shaders/reduce.wgsl';
import averagePool2DShader from '../shaders/averagePool2D.wgsl';
import gatherShader from '../shaders/gather.wgsl';
import whereShader from '../shaders/where.wgsl';
import dequantizeLinearShader from '../shaders/dequantizeLinear.wgsl';
import expandShader from '../shaders/expand.wgsl';
import padShader from '../shaders/pad.wgsl';
import convTranspose2DShader from '../shaders/convTranspose2D.wgsl';
import reLUShader from '../shaders/reLU.wgsl';
import sigmoidShader from '../shaders/sigmoid.wgsl';
import hardSwishShader from '../shaders/hardSwish.wgsl';
import hardSigmoidShader from '../shaders/hardSigmoid.wgsl';
import copyShader from '../shaders/copy.wgsl';
import batchNorm2DShader from '../shaders/batchNorm2D.wgsl';
import gELUShader from '../shaders/gELU.wgsl';
import addShader from '../shaders/add.wgsl';
import upsample2xShader from '../shaders/upsample2x.wgsl';
import concatCopyShader from '../shaders/concatCopy.wgsl';
import concat2Shader from '../shaders/concat2.wgsl';
import broadcastBinaryShader from '../shaders/broadcastBinary.wgsl';
import generalTransposeShader from '../shaders/generalTranspose.wgsl';
import splitShader from '../shaders/split.wgsl';
import profileYShader from '../shaders/profileY.wgsl';
import profileXShader from '../shaders/profileX.wgsl';
import globalAveragePoolShader from '../shaders/globalAveragePool.wgsl';
import meanHeightShader from '../shaders/meanHeight.wgsl';
import maxPool2DShader from '../shaders/maxPool2D.wgsl';
import interp1DShader from '../shaders/interp1D.wgsl';
import conv1DShader from '../shaders/conv1D.wgsl';
import spatialSoftargmaxYShader from '../shaders/spatialSoftargmaxY.wgsl';
import embeddingShader from '../shaders/embedding.wgsl';
import sDPAShader from '../shaders/sDPA.wgsl';
import crossSDPAShader from '../shaders/crossSDPA.wgsl';
import crossAttentionShader from '../shaders/crossAttention.wgsl';

export class ShaderLibrary {
  static getLinearF32Shader() {
    return linearF32Shader;
  }
  static getLinearInt8Shader() {
    return linearInt8Shader;
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
  static getLayerNormShader() {
    return layerNormShader;
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
  static getAveragePool2DShader() {
    return averagePool2DShader;
  }
  static getGatherShader() {
    return gatherShader;
  }
  static getWhereShader() {
    return whereShader;
  }
  static getDequantizeLinearShader() {
    return dequantizeLinearShader;
  }
  static getExpandShader() {
    return expandShader;
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
  static getConcat2Shader() {
    return concat2Shader;
  }
  static getBroadcastBinaryShader(binOp = "out_val = av + bv;") {
    // Substitute the per-op expression (Add/Mul/Sub/Div) into the shared template.
    return broadcastBinaryShader.replace("//__BINOP__", binOp);
  }
  static getGeneralTransposeShader() {
    return generalTransposeShader;
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
}
