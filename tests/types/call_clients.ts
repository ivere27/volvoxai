import { EngineHost, VxInferenceServiceClient, pb, type CallOptions } from '../../ts/index.js';
import { FullEngineHost, VxTrainingServiceClient, pb as fullPb } from '../../ts/full.js';

export async function callTypes(host: EngineHost, fullHost: FullEngineHost): Promise<void> {
  const inference = new VxInferenceServiceClient(host);
  const training = new VxTrainingServiceClient(fullHost);
  const options: CallOptions = { timeoutMs: 1000, signal: new AbortController().signal };
  const pending: Promise<pb.ModelHandle> = inference.loadModel(new pb.LoadModelRequest(), options);
  const model = await pending;
  const id: bigint = model.modelId;
  void id;
  // @ts-expect-error Generated ModelHandle does not contain a trainer ID.
  model.trainerId;
  // @ts-expect-error Callback overloads are not part of the call-runtime API.
  inference.loadModel(new pb.LoadModelRequest(), () => {});
  const step: fullPb.TrainStepResult = await training.trainStep(new fullPb.TrainStepRequest(), options);
  void step;
}
