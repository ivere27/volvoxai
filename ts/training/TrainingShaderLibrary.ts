import activationBackward from '../../shaders/training/activationBackward.wgsl';
import batchNorm2DBackward from '../../shaders/training/batchNorm2DBackward.wgsl';
import basicBackward from '../../shaders/training/basicBackward.wgsl';
import conv2DBackward from '../../shaders/training/conv2DBackward.wgsl';
import convTranspose2DBackward from '../../shaders/training/convTranspose2DBackward.wgsl';
import conv1DBackward from '../../shaders/training/conv1DBackward.wgsl';
import copyBackward from '../../shaders/training/copyBackward.wgsl';
import dropout from '../../shaders/training/dropout.wgsl';
import dropoutBackward from '../../shaders/training/dropoutBackward.wgsl';
import concatBackward from '../../shaders/training/concatBackward.wgsl';
import crossAttentionBackward from '../../shaders/training/crossAttentionBackward.wgsl';
import crossSdpaTraining from '../../shaders/training/crossSdpaTraining.wgsl';
import crossSdpaBackward from '../../shaders/training/crossSdpaBackward.wgsl';
import dequantizeLinearBackward from '../../shaders/training/dequantizeLinearBackward.wgsl';
import embeddingBackward from '../../shaders/training/embeddingBackward.wgsl';
import expandBackward from '../../shaders/training/expandBackward.wgsl';
import gatherBackward from '../../shaders/training/gatherBackward.wgsl';
import gatherElementsBackward from '../../shaders/training/gatherElementsBackward.wgsl';
import layerNormBackward from '../../shaders/training/layerNormBackward.wgsl';
import interp1DBackward from '../../shaders/training/interp1DBackward.wgsl';
import groupNormBackward from '../../shaders/training/groupNormBackward.wgsl';
import matMulBackward from '../../shaders/training/matMulBackward.wgsl';
import moeLinearBackward from '../../shaders/training/moeLinearBackward.wgsl';
import moeRouterBackward from '../../shaders/training/moeRouterBackward.wgsl';
import preluBackward from '../../shaders/training/preluBackward.wgsl';
import poolingBackward from '../../shaders/training/poolingBackward.wgsl';
import padBackward from '../../shaders/training/padBackward.wgsl';
import resizeBackward from '../../shaders/training/resizeBackward.wgsl';
import reduceBackward from '../../shaders/training/reduceBackward.wgsl';
import rmsNormBackward from '../../shaders/training/rmsNormBackward.wgsl';
import sdpaBackward from '../../shaders/training/sdpaBackward.wgsl';
import sdpaTraining from '../../shaders/training/sdpaTraining.wgsl';
import softmaxBackward from '../../shaders/training/softmaxBackward.wgsl';
import splitBackward from '../../shaders/training/splitBackward.wgsl';
import transposeBackward from '../../shaders/training/transposeBackward.wgsl';
import sliceBackward from '../../shaders/training/sliceBackward.wgsl';
import visionBackward from '../../shaders/training/visionBackward.wgsl';
import whereBackward from '../../shaders/training/whereBackward.wgsl';

export const TrainingShaderLibrary = Object.freeze({
  activationBackward,
  batchNorm2DBackward,
  basicBackward,
  conv2DBackward,
  convTranspose2DBackward,
  conv1DBackward,
  copyBackward,
  dropout,
  dropoutBackward,
  concatBackward,
  crossAttentionBackward,
  crossSdpaTraining,
  crossSdpaBackward,
  dequantizeLinearBackward,
  embeddingBackward,
  expandBackward,
  gatherBackward,
  gatherElementsBackward,
  layerNormBackward,
  interp1DBackward,
  groupNormBackward,
  matMulBackward,
  moeLinearBackward,
  moeRouterBackward,
  preluBackward,
  poolingBackward,
  padBackward,
  resizeBackward,
  reduceBackward,
  rmsNormBackward,
  sdpaBackward,
  sdpaTraining,
  softmaxBackward,
  sliceBackward,
  splitBackward,
  transposeBackward,
  visionBackward,
  whereBackward,
});
