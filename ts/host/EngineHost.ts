import * as pb from '../../runtime/generated/typescript/inference/volvoxai_lite.js';
import { PROTO_METHOD_RESPONSES } from '../generated/protoMethods.js';
/** Inference-profile proto host: generated C dispatch over the release WASM. */
import {
  InferenceWasmHost,
  type EngineHostOptions,
} from './InferenceWasmHost.js';

export type {
  EngineHostFetch,
  EngineHostOptions,
  ResolvedModelSource,
} from './InferenceWasmHost.js';

/** WASM CPU inference over one persistent C dispatch owner. */
export class EngineHost extends InferenceWasmHost {
  constructor(options: EngineHostOptions = {}) {
    super(options, pb, PROTO_METHOD_RESPONSES);
  }
}
