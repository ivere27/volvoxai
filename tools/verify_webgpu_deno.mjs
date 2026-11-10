#!/usr/bin/env -S deno run --no-config --unstable-webgpu -A
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

async function main() {
  const root = path.resolve(import.meta.dirname || '.');
  const bundlePath = path.resolve(root, 'dist/0.8.1/volvoxai.js');
  const wasmPath = path.resolve(root, 'dist/0.8.1/volvoxai.wasm');
  const casesDir = path.resolve(root, 'cases');

  console.log('Connecting to WebGPU adapter (high-performance)...');
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('navigator.gpu.requestAdapter returned null');
  }
  const info = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
  console.log('Device description:', info.description || info.device, '| Vendor:', info.vendor);

  const api = await import(pathToFileURL(bundlePath).href);
  const pb = api.pb;

  const diagnostics = [];
  const host = new api.FullEngineHost({
    wasmUrl: pathToFileURL(wasmPath),
    onDiagnostic: msg => {
      diagnostics.push(msg);
      console.warn('[HOST DIAGNOSTIC]', msg);
    }
  });
  const client = new api.VxInferenceServiceClient(host);

  const cases = ['packed_linear', 'packed_lstm', 'qconv2d'];
  const results = [];

  for (const caseName of cases) {
    const cDir = path.resolve(casesDir, caseName);
    const graphDoc = await fs.readFile(path.resolve(cDir, 'graph.json'));
    const weights = await fs.readFile(path.resolve(cDir, 'model.safetensors'));
    const inputBytes = await fs.readFile(path.resolve(cDir, 'x.f32'));
    const refBytes = await fs.readFile(path.resolve(cDir, 'y_ref.f32'));
    const graphObj = JSON.parse(new TextDecoder().decode(graphDoc));
    const inputShape = graphObj.inputs.x.shape;

    console.log(`\nEvaluating case [${caseName}] on WebGPU...`);

    let runtime, model, compiled, context, execution;
    try {
      runtime = await client.createRuntime(new pb.CreateRuntimeRequest());
      if (runtime.report?.status && runtime.report.status !== pb.NativeStatus.NATIVE_STATUS_OK) {
        throw new Error(`createRuntime failed: ${runtime.report.message}`);
      }

      model = await client.loadModel(new pb.LoadModelRequest({
        runtimeId: runtime.runtimeId,
        package: new pb.ModelPackage({
          graphDocument: graphDoc,
          weightShards: [new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength)]
        })
      }));
      if (model.report?.status && model.report.status !== pb.NativeStatus.NATIVE_STATUS_OK) {
        throw new Error(`loadModel failed: ${model.report.message}`);
      }

      compiled = await client.compileModel(new pb.CompileModelRequest({
        modelId: model.modelId,
        policy: new pb.BackendPolicy({
          mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
          backends: ['webgpu'],
          operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID
        })
      }));
      if (compiled.report?.status !== pb.NativeStatus.NATIVE_STATUS_OK) {
        throw new Error(`compileModel failed: ${compiled.report?.message} [code=${compiled.report?.code}]`);
      }

      const route = compiled.report.route;
      const compileOk = compiled.report.backend === 'webgpu' &&
                        route.attested === true &&
                        route.fallbackNodes === 0 &&
                        route.missingNodes === 0;

      context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId,
        graph: new pb.GraphExecution()
      }));

      execution = await client.execute(new pb.ExecuteRequest({
        contextId: context.contextId,
        inputs: new pb.InputBindings({
          values: [
            new pb.Tensor({
              name: 'x',
              shape: inputShape.map(BigInt),
              dtype: pb.DataType.DATA_TYPE_F32,
              inline: new Uint8Array(inputBytes.buffer, inputBytes.byteOffset, inputBytes.byteLength)
            })
          ]
        }),
        scope: new pb.ExecutionScope({
          entrypoint: 'evaluate',
          full: new pb.Empty()
        }),
        outputs: new pb.OutputOptions({
          storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE
        })
      }));

      if (execution.report?.status !== pb.NativeStatus.NATIVE_STATUS_OK) {
        throw new Error(`execute failed: ${execution.report?.message}`);
      }

      const outTensor = execution.outputs.find(t => t.name === 'y');
      if (!outTensor) throw new Error('Output tensor "y" missing in execution response');

      const actualBytes = outTensor.inline;
      if (actualBytes.length !== refBytes.length) {
        throw new Error(`Byte length mismatch: actual ${actualBytes.length} vs ref ${refBytes.length}`);
      }

      const actual = new Float32Array(actualBytes.buffer, actualBytes.byteOffset, actualBytes.length / 4);
      const ref = new Float32Array(refBytes.buffer, refBytes.byteOffset, refBytes.length / 4);

      let maxDiff = 0;
      let sumDiff = 0;
      let noisePower = 0;
      let sigPower = 0;
      let dot = 0, normA = 0, normR = 0;

      for (let i = 0; i < actual.length; i++) {
        const d = Math.abs(actual[i] - ref[i]);
        if (d > maxDiff) maxDiff = d;
        sumDiff += d;
        noisePower += d * d;
        sigPower += ref[i] * ref[i];
        dot += actual[i] * ref[i];
        normA += actual[i] * actual[i];
        normR += ref[i] * ref[i];
      }

      const meanDiff = sumDiff / actual.length;
      const snr = noisePower > 0 ? 10 * Math.log10(sigPower / noisePower) : 999.0;
      const cosine = (normA > 0 && normR > 0) ? dot / (Math.sqrt(normA) * Math.sqrt(normR)) : 0.0;
      const passed = maxDiff < 1e-4 || snr > 50.0;

      const record = {
        case: caseName,
        backend: 'webgpu',
        status: passed ? 'PASSED' : 'ACCURACY_FAIL',
        max_err: maxDiff,
        mean_err: meanDiff,
        snr_db: snr,
        cosine: cosine,
        route_attested: route.attested,
        active_nodes: route.activeNodes,
        selected_nodes: route.selectedNodes
      };
      results.push(record);
      console.log(`[${record.status}] Case: ${caseName} | MaxErr: ${maxDiff.toExponential(4)} | SNR: ${snr.toFixed(2)} dB | Cosine: ${cosine.toFixed(6)}`);

    } catch (err) {
      console.error(`[FAILED] Case: ${caseName} | Error:`, err.message);
      results.push({
        case: caseName,
        backend: 'webgpu',
        status: 'FAILED',
        error: err.message
      });
    } finally {
      if (context) await client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId: context.contextId })).catch(() => {});
      if (compiled) await client.releaseCompiledModel(new pb.CompiledModelRef({ compiledModelId: compiled.compiledModelId })).catch(() => {});
      if (model) await client.releaseModel(new pb.ModelRef({ modelId: model.modelId })).catch(() => {});
      if (runtime) await client.releaseRuntime(new pb.RuntimeRef({ runtimeId: runtime.runtimeId })).catch(() => {});
    }
  }

  const summary = {
    format: 'volvox-webgpu-deno-verification-v1',
    device: info.description || info.device,
    vendor: info.vendor,
    total: results.length,
    passed: results.filter(r => r.status === 'PASSED').length,
    results
  };

  await fs.writeFile(path.resolve(root, 'webgpu_verification_results.json'), JSON.stringify(summary, null, 2));
  console.log(`\nWebGPU Verification Summary: ${summary.passed}/${summary.total} tests PASSED.`);

  if (summary.passed !== summary.total) {
    Deno.exit(1);
  }
}

main().catch(err => {
    console.error('Fatal error:', err);
    Deno.exit(1);
});
