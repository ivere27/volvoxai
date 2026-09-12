import { reportTransport, checkedReport } from '../../../tools/proto_report_fixture.mjs';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';

const bytes = values => new Uint8Array(values.buffer,values.byteOffset,values.byteLength).slice();
const values = tensor => new Float32Array(tensor.inline.buffer,tensor.inline.byteOffset,tensor.inline.byteLength/4).slice();
function close(actual,expected,label) {
  assert.equal(actual.length,expected.length,label);
  for(let i=0;i<actual.length;i++)assert.ok(Math.abs(actual[i]-expected[i])<3e-6+2e-5*Math.abs(expected[i]),`${label}[${i}] ${actual[i]} != ${expected[i]}`);
}

// Independent dense LoRA + mean cross-entropy derivative, with no C backward
// or numerical training oracle shared by the implementation under test.
function derivative(x,base,bias,A,B,scale,transpose) {
  const hidden=new Float64Array(4),logits=new Float64Array(4),dy=new Float64Array(4);
  const gradA=new Float64Array(6),gradB=new Float64Array(4);
  let loss=0;
  for(let row=0;row<2;row++) {
    for(let k=0;k<2;k++)for(let d=0;d<3;d++)hidden[row*2+k]+=x[row*3+d]*A[d*2+k];
    for(let out=0;out<2;out++) {
      logits[row*2+out]=bias?.[out]??0;
      for(let d=0;d<3;d++)logits[row*2+out]+=x[row*3+d]*base[transpose?out*3+d:d*2+out];
      for(let k=0;k<2;k++)logits[row*2+out]+=scale*hidden[row*2+k]*B[k*2+out];
    }
    const maximum=Math.max(logits[row*2],logits[row*2+1]);
    const sum=Math.exp(logits[row*2]-maximum)+Math.exp(logits[row*2+1]-maximum);
    loss+=(Math.log(sum)+maximum-logits[row*2+row])/2;
    for(let out=0;out<2;out++)dy[row*2+out]=(Math.exp(logits[row*2+out]-maximum)/sum-(out===row?1:0))/2;
  }
  for(let k=0;k<2;k++)for(let out=0;out<2;out++)for(let row=0;row<2;row++)
    gradB[k*2+out]+=scale*hidden[row*2+k]*dy[row*2+out];
  for(let d=0;d<3;d++)for(let k=0;k<2;k++)for(let row=0;row<2;row++)for(let out=0;out<2;out++)
    gradA[d*2+k]+=scale*x[row*3+d]*B[k*2+out]*dy[row*2+out];
  return {loss,gradA,gradB};
}

export async function qualifyLora(api,wasmUrl,backends=['wasm']) {
  const p=api.pb,check = checkedReport,F32=p.DataType.DATA_TYPE_F32,I8=p.DataType.DATA_TYPE_I8;
  const results=[];
  let refusalCases=0,conversionCases=0,gradientValues=0;
  for(const transpose of [false,true])for(const nonzeroB of [false,true]) {
    let graph,shard;
    const host=new api.FullEngineHost({wasmUrl});
    const plan=new api.VxPlanningServiceClient(reportTransport(host)),training=new api.VxTrainingServiceClient(reportTransport(host));
    const inference=new api.VxInferenceServiceClient(reportTransport(host)),quantization=new api.VxQuantizationServiceClient(reportTransport(host));
    const makeTensor=(name,shape,values,dtype=F32)=>new p.Tensor({name,shape:shape.map(BigInt),dtype,inline:bytes(values)});
    const base=Float32Array.of(.4,-.25,.1,.3,-.6,.2),bias=nonzeroB?Float32Array.of(.15,-.05):null;
    const x=Float32Array.of(.5,-1,.25,-.7,.2,.9);
    try {
      const original=check(await plan.createGraphPlan(new p.CreateGraphPlanRequest({graph:new p.GraphPlanningSource({
        definition:new p.GraphDefinition({inputs:[new p.GraphTensorDefinition({name:'x',dtype:F32,
          shape:[2,3].map(d=>new p.GraphAxis({fixedExtent:BigInt(d)}))})],outputs:['x']}),
        weights:[new p.PlanningWeight({name:'base',dtype:F32,shape:(transpose?[2,3]:[3,2]).map(BigInt)}),
          ...(bias?[new p.PlanningWeight({name:'bias',dtype:F32,shape:[2n]})]:[])],
      })})));
      const loraRequest={graphPlanId:original.graphPlanId,inputName:'x',baseWeightName:'base',
        ...(bias?{biasName:'bias'}:{}),rank:2,transposeBaseWeight:transpose,
        baseOperator:nonzeroB?'MatMul':'Linear',name:'adapter',selectOutput:true,
        ...(nonzeroB?{scale:1.75,bInitializer:new p.TensorInitializer({xavierUniform:new p.XavierInitializer(),seed:39})}:{alpha:3}),
      };
      const authored=check(await training.buildLoraLinear(new p.BuildLoraLinearRequest(loraRequest)));
      assert.deepEqual(authored.trainableNames,['adapter.lora_a','adapter.lora_b']);
      assert.equal(authored.scale,nonzeroB?1.75:1.5);
      assert.equal(authored.plan.nodes.length,5);
      assert.equal(check(await plan.getGraphPlan(new p.GraphPlanRef(original))).plan.graphFingerprint,original.plan.graphFingerprint);
      const initialized=Object.fromEntries(authored.initializedWeights.map(t=>[t.name,values(t)]));
      assert.deepEqual(initialized['adapter.lora_scale'],Float32Array.of(authored.scale));
      if(!nonzeroB)assert.deepEqual(initialized['adapter.lora_b'],new Float32Array(4));
      graph=check(await plan.exportGraphPlan(new p.GraphPlanRef(authored))).source.graphDocument;
      const document=JSON.parse(new TextDecoder().decode(graph));
      assert.deepEqual(document.outputs,['adapter.out']);
      assert.equal(document.nodes[0].params.weight_layout,transpose?'dout_din':'din_dout');
      shard=check(await plan.writeSafetensors(new p.WriteSafetensorsRequest({edits:[
        new p.SafetensorsEdit({setTensor:makeTensor('base',transpose?[2,3]:[3,2],base)}),
        ...(bias?[new p.SafetensorsEdit({setTensor:makeTensor('bias',[2],bias)})]:[]),
        ...authored.initializedWeights.map(t=>new p.SafetensorsEdit({setTensor:t})),
      ]}))).data;
      for(const change of [{rank:0},{alpha:0},{alpha:NaN},{baseOperator:'Conv2D'},{inputName:'missing'},
        {graphPlanId:authored.graphPlanId}]) {
        const invalid={...loraRequest,...change};
        if(Object.hasOwn(change,'alpha'))delete invalid.scale;
        const failed=await training.buildLoraLinear(new p.BuildLoraLinearRequest(invalid));
        assert.notEqual(failed.report.status,0);assert.equal(failed.graphPlanId,0n);assert.equal(failed.initializedWeights.length,0);refusalCases++;
      }
      const runtime=check(await inference.createRuntime(new p.CreateRuntimeRequest()));
      const model=check(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,
        package:new p.ModelPackage({graphDocument:graph,weightShards:[shard]})})));
      const trained=[];
      for(const backend of backends) {
        const trainer=check(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend})));
        const beforeInputFailure=check(await training.getTrainerState(new p.TrainerRef(trainer)));
        const inputFailure=await training.trainStep(new p.TrainStepRequest({trainerId:trainer.trainerId,
          inputs:[makeTensor('x',[2,3],x,p.DataType.DATA_TYPE_I32)],
          losses:[new p.CrossEntropyLoss({name:'classification',logitsName:authored.outputName,targets:[0,1]})],
          trainableNames:authored.trainableNames,
          optimizer:new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,learningRate:.05}),
        }));
        assert.equal(inputFailure.report.inputIssue.code,p.InputValidationCode.INPUT_VALIDATION_CODE_DTYPE_MISMATCH);
        assert.equal(inputFailure.report.inputIssue.expected.dtype,F32);
        assert.equal(check(await training.getTrainerState(new p.TrainerRef(trainer))).optimizerStep,beforeInputFailure.optimizerStep);
        let A=initialized['adapter.lora_a'].slice(),B=initialized['adapter.lora_b'].slice();
        let saved;
        for(let step=0;step<3;step++) {
          const expected=derivative(x,base,bias,A,B,authored.scale,transpose);
          let actual=check(await training.trainStep(new p.TrainStepRequest({trainerId:trainer.trainerId,
            inputs:[makeTensor('x',[2,3],x)],losses:[new p.CrossEntropyLoss({name:'classification',
              logitsName:authored.outputName,targets:[0,1]})],trainableNames:authored.trainableNames,
            optimizer:new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,learningRate:.05}),
          })));
          const deadline=Date.now()+30000;
          while(actual.state===p.ResultState.RESULT_STATE_PENDING) {
            assert.ok(Date.now()<deadline);await new Promise(resolve=>setTimeout(resolve,1));
            actual=check(await training.getTrainStep(new p.TrainStepRef({trainerId:trainer.trainerId,microbatchId:actual.microbatchId})));
          }
          assert.equal(actual.state,p.ResultState.RESULT_STATE_READY);assert.ok(actual.updateApplied);
          close([actual.loss],[expected.loss],`${backend} loss`);
          saved=check(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:trainer.trainerId}))).shards[0];
          const after=check(await plan.readSafetensors(new p.ReadSafetensorsRequest({source:saved})));
          const weights=Object.fromEntries(after.tensors.map(t=>[t.name,values(t)]));
          close(weights['adapter.lora_a'],A.map((v,i)=>Math.fround(v-Math.fround(.05)*expected.gradA[i])),`${backend} grad A`);
          close(weights['adapter.lora_b'],B.map((v,i)=>Math.fround(v-Math.fround(.05)*expected.gradB[i])),`${backend} grad B`);
          gradientValues+=A.length+B.length;
          assert.deepEqual(weights.base,base);if(bias)assert.deepEqual(weights.bias,bias);
          assert.deepEqual(weights['adapter.lora_scale'],Float32Array.of(authored.scale));
          A=weights['adapter.lora_a'];B=weights['adapter.lora_b'];
        }
        trained.push({A,B});
        // Exporting a quantized inference snapshot keeps the persistent F32
        // masters, optimizer step and original Model unchanged.
        const packed=check(await quantization.quantizeWeight(new p.QuantizeWeightRequest({
          weight:makeTensor('adapter.lora_a',[3,2],A),transposeSource:true,
        })));
        assert.deepEqual(packed.weight.shape,[2n,3n]);assert.equal(packed.weight.dtype,I8);
        const output=new Int8Array(packed.weight.inline.buffer,packed.weight.inline.byteOffset,6);
        for(let row=0;row<2;row++) {
          const maximum=Math.max(...Array.from({length:3},(_,column)=>Math.abs(A[column*2+row])));
          assert.equal(packed.scales[row],Math.fround(maximum/127));
          for(let column=0;column<3;column++)assert.ok(Math.abs(output[row*3+column]*packed.scales[row]-A[column*2+row])<=packed.scales[row]*.501);
        }
        const dequantized=check(await quantization.dequantizeWeight(new p.DequantizeWeightRequest({
          weight:packed.weight,scales:packed.scales,transposeDestination:true,
        })));
        assert.deepEqual(dequantized.tensor.shape,[3n,2n]);
        const restored=values(dequantized.tensor);
        for(let row=0;row<2;row++)for(let column=0;column<3;column++)
          assert.equal(restored[column*2+row],Math.fround(output[row*3+column]*packed.scales[row]));
        const scaleB=[.125,.25];
        const makeTemplate=async tensors=>check(await plan.writeSafetensors(new p.WriteSafetensorsRequest({
          edits:tensors.map(t=>new p.SafetensorsEdit({setTensor:t})),
          metadata:new p.SafetensorsMetadata({entries:[new p.SafetensorsMetadataEntry({key:'fixture',value:'template µ'})]}),
        }))).data;
        const template=[await makeTemplate([
          makeTensor('qa',[2,3],new Int8Array(6),I8),makeTensor('sa',[2],Float32Array.of(.1,.1)),
          makeTensor('za',[2],new Int8Array(2),I8),makeTensor('unchanged',[3],Uint16Array.of(7,11,65535),p.DataType.DATA_TYPE_U16),
        ]),await makeTemplate([
          makeTensor('qb',[2,2],new Int8Array(4),I8),makeTensor('sb',[2],Float32Array.from(scaleB)),
          makeTensor('zb',[2],new Int8Array(2),I8),
        ])];
        const bindings=[new p.TrainerQuantizedWeightBinding({masterName:'adapter.lora_a',weightName:'qa',
          scaleName:'sa',zeroPointName:'za',transposeMaster:true}),new p.TrainerQuantizedWeightBinding({
          masterName:'adapter.lora_b',weightName:'qb',scaleName:'sb',zeroPointName:'zb',preserveScales:true})];
        const exportRequest={trainerId:trainer.trainerId,templateShards:template,bindings};
        const untouched=template.map(t=>t.slice());
        const snapshot=check(await quantization.exportQuantizedTrainerWeights(new p.ExportQuantizedTrainerWeightsRequest(exportRequest)));
        assert.equal(snapshot.report.lineage.modelId,model.modelId);
        assert.equal(snapshot.shards.length,2);assert.deepEqual(template,untouched);
        const templates=await Promise.all(snapshot.shards.map(async source=>(plan.readSafetensors(new p.ReadSafetensorsRequest({source}))).then(check)));
        for(const content of templates)assert.deepEqual(content.metadata.entries.map(e=>[e.key,e.value]),[['fixture','template µ']]);
        const projected=Object.fromEntries(templates.flatMap(content=>content.tensors).map(t=>[t.name,t]));
        assert.deepEqual(projected.qa.inline,packed.weight.inline);assert.deepEqual(values(projected.sa),Float32Array.from(packed.scales));
        assert.deepEqual(projected.unchanged.inline,bytes(Uint16Array.of(7,11,65535)));
        assert.deepEqual(values(projected.sb),Float32Array.from(scaleB));
        const packedB=check(await quantization.quantizeWeight(new p.QuantizeWeightRequest({weight:makeTensor('b',[2,2],B),
          preserve:new p.PreservedWeightScales({scales:scaleB})})));
        assert.deepEqual(projected.qb.inline,packedB.weight.inline);
        assert.equal(snapshot.saturationCount,packed.saturationCount+packedB.saturationCount);
        assert.deepEqual(check(await quantization.exportQuantizedTrainerWeights(new p.ExportQuantizedTrainerWeightsRequest(exportRequest))).shards,snapshot.shards);
        const badZero=check(await plan.writeSafetensors(new p.WriteSafetensorsRequest({source:template[1],
          edits:[new p.SafetensorsEdit({setTensor:makeTensor('zb',[2],Int8Array.of(1,0),I8)})]}))).data;
        for(const change of [{bindings:[]},{templateShards:[]},{bindings:[bindings[0],bindings[0]]},
          {bindings:[bindings[0],new p.TrainerQuantizedWeightBinding({...bindings[1],masterName:'missing'})]},
          {templateShards:[template[0],badZero]},{templateShards:[...template,template[0]]}]) {
          const refused=await quantization.exportQuantizedTrainerWeights(new p.ExportQuantizedTrainerWeightsRequest({...exportRequest,...change}));
          assert.notEqual(refused.report.status,0);assert.equal(refused.shards.length,0);assert.equal(refused.saturationCount,0n);refusalCases++;
        }
        assert.deepEqual(template,untouched);
        conversionCases++;
        assert.deepEqual(check(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:trainer.trainerId}))).shards[0],saved);
        conversionCases++;
        check(await training.releaseTrainer(new p.TrainerRef(trainer)));
      }
      for(let i=1;i<trained.length;i++) {close(trained[i].A,trained[0].A,'backend A');close(trained[i].B,trained[0].B,'backend B');}
      // Ties-to-even, saturation and the zero-row scale policy have explicit
      // expected integers independent of the implementation.
      const edge=check(await quantization.quantizeWeight(new p.QuantizeWeightRequest({
        weight:makeTensor('edge',[2,4],Float32Array.of(.5,1.5,2.5,128,-.5,-1.5,-2.5,-129)),
        preserve:new p.PreservedWeightScales({scales:[1,1]}),
      })));
      assert.deepEqual(new Int8Array(edge.weight.inline.buffer),Int8Array.of(0,2,2,127,0,-2,-2,-127));
      assert.equal(edge.saturationCount,2n);assert.deepEqual(edge.scales,[1,1]);
      const zero=check(await quantization.quantizeWeight(new p.QuantizeWeightRequest({weight:makeTensor('zero',[2,3],new Float32Array(6))})));
      assert.deepEqual(zero.scales,[1,1]);assert.deepEqual(zero.weight.inline,new Uint8Array(6));conversionCases+=2;
      for(const request of [{}, {weight:makeTensor('bad',[2,3],Float32Array.of(NaN,0,0,0,0,0))},
        {weight:makeTensor('bad',[2,3],new Float32Array(6)),preserve:new p.PreservedWeightScales({scales:[0,1]})},
        {weight:makeTensor('bad',[2,3],new Float32Array(6)),preserve:new p.PreservedWeightScales({scales:[1]})},
      ]) {const failed=await quantization.quantizeWeight(new p.QuantizeWeightRequest(request));assert.notEqual(failed.report.status,0);assert.equal(failed.weight,undefined);refusalCases++;}
      assert.notEqual((await quantization.dequantizeWeight(new p.DequantizeWeightRequest({weight:zero.weight,scales:[1,Infinity]}))).report.status,0);refusalCases++;
      results.push({transpose,nonzeroB,backends,status:'pass'});
    } finally {await host.close();}
  }
  const routed=await qualifyRouted(api,wasmUrl,backends);
  return {status:'pass',cases:results,gradientValues,conversionCases,refusalCases,routed};
}

async function qualifyRouted(api,wasmUrl,backends) {
  const p=api.pb,check = checkedReport,F32=p.DataType.DATA_TYPE_F32,results=[];
  for(const dropout of [0,.25])for(const withBias of [false,true]) {
    let graph,shard;
    const host=new api.FullEngineHost({wasmUrl,fetch:async url=>new Response(String(url).endsWith('graph.json')?graph:shard)});
    try {
      const plan=new api.VxPlanningServiceClient(reportTransport(host)),training=new api.VxTrainingServiceClient(reportTransport(host)),inference=new api.VxInferenceServiceClient(reportTransport(host));
      const shapes={x:[2,3],indices:[2,2],routes:[2,2],down:[2,3,2],up:[2,2,3],...(withBias?{downBias:[2,2],upBias:[2,3]}:{})};
      const weightNames=Object.keys(shapes).slice(3);
      const initial=Object.fromEntries(weightNames.map((name,k)=>[name,Float32Array.from({length:shapes[name].reduce((a,b)=>a*b,1)},(_,i)=>Math.sin(i+k+.7)*.2)]));
      const make=(name,values)=>new p.Tensor({name,shape:shapes[name].map(BigInt),dtype:F32,inline:bytes(values)});
      const source=check(await plan.createGraphPlan(new p.CreateGraphPlanRequest({graph:new p.GraphPlanningSource({
        definition:new p.GraphDefinition({inputs:['x','indices','routes'].map(name=>new p.GraphTensorDefinition({name,dtype:F32,
          shape:shapes[name].map(d=>new p.GraphAxis({fixedExtent:BigInt(d)}))})),outputs:['x']}),
        weights:weightNames.map(name=>new p.PlanningWeight({name,dtype:F32,shape:shapes[name].map(BigInt)})),
      })})));
      const request={graphPlanId:source.graphPlanId,inputName:'x',downWeightName:'down',upWeightName:'up',
        routeIndicesName:'indices',routeWeightsName:'routes',...(withBias?{downBiasName:'downBias',upBiasName:'upBias'}:{}),
        dropout,seed:31,name:'adapter',selectOutput:true};
      const built=check(await training.buildRoutedAdapter(new p.BuildRoutedAdapterRequest(request)));
      graph=check(await plan.exportGraphPlan(new p.GraphPlanRef(built))).source.graphDocument;
      const document=JSON.parse(new TextDecoder().decode(graph));
      assert.deepEqual(document.nodes.map(n=>n.opType),['MoELinear','GELU','MoELinear',...(dropout?['Dropout']:[]),'Add']);
      assert.deepEqual(document.outputs,['adapter.out']);
      assert.deepEqual(document.nodes[0].outputs.out.shape,[2,2]);
      assert.deepEqual(document.nodes.at(-1).outputs.out.shape,[2,3]);
      if(dropout)assert.deepEqual(document.nodes[3].params,{ratio:dropout,seed:31});
      for(const change of [{dropout:1},{dropout:NaN},{routeWeightsName:'missing'},{upWeightName:'down'},{graphPlanId:built.graphPlanId}]) {
        const refused=await training.buildRoutedAdapter(new p.BuildRoutedAdapterRequest({...request,...change}));
        assert.notEqual(refused.report.status,0);assert.equal(refused.graphPlanId,0n);
      }
      assert.equal(check(await plan.getGraphPlan(new p.GraphPlanRef(source))).plan.graphFingerprint,source.plan.graphFingerprint);
      shard=check(await plan.writeSafetensors(new p.WriteSafetensorsRequest({edits:weightNames.map(name=>new p.SafetensorsEdit({setTensor:make(name,initial[name])}))}))).data;
      const runtime=check(await inference.createRuntime(new p.CreateRuntimeRequest()));
      const model=check(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']})));
      const outputs=[];
      for(const backend of backends) {
        const trainer=check(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend,rngSeed:71n})));
        let output;
        for(let step=0;step<3;step++) {
          const start={trainerId:trainer.trainerId,inputs:[make('x',Float32Array.of(.4,-.3,.6,-.1,.2,-.7)),
            make('indices',Float32Array.of(0,1,1,0)),make('routes',Float32Array.of(.7,.3,.4,.6))],
            losses:[new p.CrossEntropyLoss({name:'ce',logitsName:built.outputName,targets:[0,2]})],trainableNames:weightNames,
            optimizer:new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,learningRate:.03})};
          output=check(await training.trainStep(new p.TrainStepRequest(start)));
          const deadline=Date.now()+30000;
          while(output.state===p.ResultState.RESULT_STATE_PENDING) {
            assert.ok(Date.now()<deadline);await new Promise(resolve=>setTimeout(resolve,1));
            output=check(await training.getTrainStep(new p.TrainStepRef({trainerId:trainer.trainerId,microbatchId:output.microbatchId})));
          }
          assert.equal(output.state,p.ResultState.RESULT_STATE_READY);assert.ok(output.updateApplied);
        }
        const saved=check(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:trainer.trainerId}))).shards[0];
        const after=check(await plan.readSafetensors(new p.ReadSafetensorsRequest({source:saved})));
        const current=Object.fromEntries(after.tensors.map(t=>[t.name,values(t)]));
        assert.notDeepEqual(current.down,initial.down);
        outputs.push({current,loss:output.loss});
        check(await training.releaseTrainer(new p.TrainerRef(trainer)));
      }
      for(const output of outputs.slice(1)) {
        close([output.loss],[outputs[0].loss],'routed loss');
        for(const name of weightNames)close(output.current[name],outputs[0].current[name],'routed '+name);
      }
      results.push({dropout,withBias,backends,status:'pass'});
    } finally {await host.close();}
  }
  return results;
}

if(process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
  const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href);
  const result=await qualifyLora(api,pathToFileURL(path.resolve(args.get('--wasm'))),(args.get('--backends')??'wasm').split(','));
  console.log(JSON.stringify(result));
  if(args.has('--out'))await writeFile(args.get('--out'),`${JSON.stringify(result,null,2)}\n`);
}
