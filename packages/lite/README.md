# volvoxai-lite

VolvoxAI's standalone WASM CPU inference package for browsers and Node.js.
It includes the readable and minified inference JavaScript bundles and their
WASM companion. It has no dependency on the full `volvoxai` package.

```sh
npm install volvoxai-lite
```

```js
import { EngineHost, VxPlatformServiceClient, pb } from 'volvoxai-lite';

const host = new EngineHost();
try {
  const platform = new VxPlatformServiceClient(host);
  const result = await platform.getPlatformInfo(new pb.Empty());
  console.log(result);
} finally {
  await host.close();
}
```

Use `volvoxai-lite/min` for the minified entry. Both entries load the sibling
`volvoxai.lite.wasm` by default. Browser bundlers must emit that companion asset;
you can also supply `new EngineHost({ wasmUrl: '/assets/volvoxai.lite.wasm' })`.
The WASM file is exported as `volvoxai-lite/volvoxai.lite.wasm`.

The package exposes inference, tensor buffers, text/tokenization, scheduling and
profiling through the generated protobuf API. WebGPU, training, QAT and PTQ
authoring are available in the full `volvoxai` package. Already-quantized models
supported by the WASM inference profile can run in `volvoxai-lite`.

Both packages use the same release version and engine API. See the
[quickstart](https://github.com/ivere27/volvoxai/blob/master/docs/quickstart.md)
and [API guide](https://github.com/ivere27/volvoxai/blob/master/docs/api-discovery.md).

MIT licensed; see `LICENSE` in this package.
