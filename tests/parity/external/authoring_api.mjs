import { reportTransport, checkedReport } from '../../../tools/proto_report_fixture.mjs';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const encoder = new TextEncoder(), decoder = new TextDecoder();
const packed = values => new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice();

// Independent format oracle: no runtime parser or serializer is involved.
function unpack(bytes) {
  const headerSize = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  assert.equal(headerSize % 8, 0);
  const header = JSON.parse(decoder.decode(bytes.subarray(8, 8 + headerSize)));
  const tensors = Object.entries(header).filter(([name]) => name !== '__metadata__').map(([name, info]) => ({
    name, ...info, bytes: bytes.slice(8 + headerSize + info.data_offsets[0], 8 + headerSize + info.data_offsets[1]),
  }));
  let cursor = 0;
  for (const tensor of tensors) { assert.equal(tensor.data_offsets[0], cursor); cursor = tensor.data_offsets[1]; }
  assert.equal(8 + headerSize + cursor, bytes.length);
  return { tensors, metadata: header.__metadata__ };
}

export async function qualifyAuthoring(api, legacyInitialize, wasmUrl) {
  const p = api.pb, check = checkedReport;
  const host = new (api.FullEngineHost ?? api.EngineHost)({ wasmUrl });
  const foreignHost = new (api.FullEngineHost ?? api.EngineHost)({ wasmUrl });
  const planning = new api.VxPlanningServiceClient(reportTransport(host));
  const foreign = new api.VxPlanningServiceClient(reportTransport(foreignHost));
  const F32 = p.DataType.DATA_TYPE_F32;
  const axis = value => new p.GraphAxis(typeof value === 'string' ? {dimension:value} : {fixedExtent:BigInt(value)});
  const spec = (name, shape=['B',3], dtype=F32) => new p.GraphTensorDefinition({name, dtype, shape:shape.map(axis)});
  const node = (id, operatorName, input, output, parameters=[]) => new p.GraphNodeDefinition({
    ...(id === undefined ? {} : {id}), operatorName,
    inputs:[new p.GraphInputBinding({port:'input', tensorName:input})],
    outputs:[new p.GraphOutputDefinition({port:'out', tensor:spec(output)})], parameters,
  });
  const edit = properties => new p.GraphEdit(properties);
  const planRef = value => new p.GraphPlanRef({graphPlanId:value.graphPlanId ?? value});
  const retained = [];
  const keep = value => { const got = check(value); retained.push(got.graphPlanId); return got; };
  let transactions=0, refusals=0, storageCases=0, initializerCases=0, initializerValues=0, initializerRoundedDifferences=0;
  try {
    const definition = new p.GraphDefinition({
      dimensions:[new p.GraphDimension({name:'B',minimum:1n,maximum:8n})],
      inputs:[spec('x')],
      // Reserve node_0 even though it occurs later in the definition.
      nodes:[node(undefined,'Identity','x','hidden'),node('node_0','Tanh','hidden','y')], outputs:['y'],
    });
    const source = new p.GraphPlanningSource({definition});
    const serialized=check(await planning.serializeGraph(definition));
    const original = keep(await planning.createGraphPlan(new p.CreateGraphPlanRequest({graph:source})));
    const originalFingerprint = original.plan.graphFingerprint;
    assert.deepEqual(original.plan.nodes.map(n=>n.id),['node_1','node_0']);
    definition.inputs[0].name = 'changed-after-publication';
    const exported = check(await planning.exportGraphPlan(planRef(original)));
    const document = JSON.parse(decoder.decode(exported.source.graphDocument));
    assert.deepEqual(serialized.data,exported.source.graphDocument);
    assert.equal(document.inputs.x.dtype,'float32');
    assert.equal(document.dimensions.B.multiple_of,1);
    assert.equal(exported.source.definition,undefined);
    keep(await planning.createGraphPlan(new p.CreateGraphPlanRequest({graph:exported.source})));
    assert.notEqual((await foreign.exportGraphPlan(planRef(original))).report.status,0); refusals++;

    const transaction = keep(await planning.editGraphPlan(new p.EditGraphPlanRequest({graphPlanId:original.graphPlanId,edits:[
      edit({removeNode:'node_1'}),
      edit({removeInput:'x'}),
      edit({setInput:spec('z')}),
      edit({replaceNode:new p.GraphNodeReplacement({id:'node_0',node:node(undefined,'Tanh','z','result')})}),
      edit({selectOutputs:new p.GraphOutputSelection({names:['result']})}),
      edit({setDimension:new p.GraphDimension({name:'B',minimum:2n,maximum:6n,multipleOf:2n})}),
      edit({setDimension:new p.GraphDimension({name:'unused',minimum:1n,maximum:1n})}),
      edit({removeDimension:'unused'}), edit({removeWeight:'absent'}), edit({removeNode:'absent'}),
    ]})));
    transactions++;
    assert.deepEqual(transaction.plan.nodes.map(n=>n.id),['node_0']);
    assert.equal(transaction.plan.dimensions[0].minimum,2n);
    assert.deepEqual(check(await planning.getGraphPlan(planRef(original))).plan.graphFingerprint,originalFingerprint);
    assert.ok(!transaction.plan.tensors.some(t=>t.name==='x'));

    const f16 = new p.PlanningWeight({name:'half',dtype:p.DataType.DATA_TYPE_F16,shape:[3n]});
    const weighted = keep(await planning.editGraphPlan(new p.EditGraphPlanRequest({graphPlanId:transaction.graphPlanId,edits:[
      edit({setWeight:f16}),
      edit({addNode:node(undefined,'Clip','result','clipped',[
        new p.GraphParameter({name:'min',value:new p.NodeParameterValue({numberValue:-.5})}),
        new p.GraphParameter({name:'max',value:new p.NodeParameterValue({numberValue:.5})}),
      ])}), edit({selectOutputs:new p.GraphOutputSelection({names:['clipped']})}),
    ]})));
    transactions++;
    const weightedSource = check(await planning.exportGraphPlan(planRef(weighted))).source;
    assert.equal(weightedSource.weights.find(w=>w.name==='half').dtype,p.DataType.DATA_TYPE_F16);
    assert.equal(weighted.plan.nodes.at(-1).id,'node_1');
    keep(await planning.createGraphPlan(new p.CreateGraphPlanRequest({graph:weightedSource})));
    const unweighted = keep(await planning.editGraphPlan(new p.EditGraphPlanRequest({graphPlanId:weighted.graphPlanId,edits:[
      edit({setWeight:new p.PlanningWeight({name:'half',dtype:F32,shape:[2n]})}),edit({removeWeight:'half'}),
    ]})));
    assert.equal(check(await planning.exportGraphPlan(planRef(unweighted))).source.weights.length,0); transactions++;

    for (const edits of [
      [edit({removeInput:'z'})], [edit({removeDimension:'B'})],
      [edit({replaceNode:new p.GraphNodeReplacement({id:'missing',node:node(undefined,'Tanh','z','result')})})],
      [edit({selectOutputs:new p.GraphOutputSelection()})],
      [edit({setWeight:new p.PlanningWeight({name:'bad',dtype:F32,shape:[-1n]})})],
      [edit({addNode:node('node_0','Identity','result','duplicate')})], [edit({})],
    ]) {
      const failed=await planning.editGraphPlan(new p.EditGraphPlanRequest({graphPlanId:transaction.graphPlanId,edits}));
      assert.notEqual(failed.report.status,0); assert.equal(failed.graphPlanId,0n); refusals++;
      assert.equal(check(await planning.getGraphPlan(planRef(original))).plan.graphFingerprint,originalFingerprint);
    }
    const invalidDefinition = new p.GraphDefinition({inputs:[spec('same',[1]),spec('same',[1])],outputs:['same']});
    const bad = await planning.createGraphPlan(new p.CreateGraphPlanRequest({graph:new p.GraphPlanningSource({definition:invalidDefinition})}));
    assert.notEqual(bad.report.status,0); assert.equal(bad.graphPlanId,0n); refusals++;

    // Quantization references, values and F16 storage descriptors survive a
    // complete edit/export/recreate cycle without a GPU or a model payload.
    const qDefinition=new p.GraphDefinition({inputs:[spec('q',[2],p.DataType.DATA_TYPE_I8)],outputs:['q'],
      quantization:[new p.GraphQuantizationReference({tensorName:'q',scaleTensor:'scale',zeroPointTensor:'zero'})]});
    // Metadata-only authoring does not need numeric scale/zero payloads.
    const qSerialized=check(await planning.serializeGraph(qDefinition));
    const quantization=new p.TensorAffineQuantization({tensorName:'q',parameters:new p.AffineQuantizationParameters({
      perTensor:new p.PerTensorAffineQuantization({scale:.25,zeroPoint:-3}),
    })});
    const quantized=keep(await planning.createGraphPlan(new p.CreateGraphPlanRequest({graph:new p.GraphPlanningSource({
      definition:qDefinition,weights:[new p.PlanningWeight({name:'scale',dtype:F32,shape:[1n]}),
        new p.PlanningWeight({name:'zero',dtype:p.DataType.DATA_TYPE_I8,shape:[1n]})],quantization:[quantization],
    })})));
    const qExport=check(await planning.exportGraphPlan(planRef(quantized)));
    assert.deepEqual(qSerialized.data,qExport.source.graphDocument);
    assert.equal(qExport.source.quantization[0].parameters.perTensor.scale,.25);
    keep(await planning.createGraphPlan(new p.CreateGraphPlanRequest({graph:qExport.source})));
    keep(await planning.editGraphPlan(new p.EditGraphPlanRequest({graphPlanId:quantized.graphPlanId,edits:[
      edit({replaceQuantization:new p.GraphQuantizationReplacement({references:qDefinition.quantization,values:[quantization]})}),
    ]}))); transactions++;

    const write = async request => (await planning.writeSafetensors(new p.WriteSafetensorsRequest(request)));
    const read = async request => (await planning.readSafetensors(new p.ReadSafetensorsRequest(request)));
    const tensor = (name,dtype,shape,bytes) => new p.Tensor({name,dtype,shape:shape.map(BigInt),...(bytes===undefined?{}:{inline:bytes})});
    const set = t => new p.SafetensorsEdit({setTensor:t});
    const widths=[1,4,6,6,8,8,8,8,8,8,8,16,16,16,16,32,32,32,64,64,64,64];
    const entries=widths.map((bits,index)=>tensor(`dtype_${index+1}`,index+1,[8],Uint8Array.from({length:bits},(_,i)=>(index*31+i*7)&255)));
    // BOOL uses a whole byte in SafeTensors, not a packed single bit.
    entries[0]=tensor('dtype_1',1,[8],Uint8Array.of(0,1,1,0,0,1,0,1));
    const longName='name_'.repeat(50)+'\u0000한글';
    entries.push(tensor(longName,F32,Array(12).fill(1),packed(Float32Array.of(3.25))));
    entries.push(tensor('__proto__',F32,[0,9007199254740991n],new Uint8Array()));
    entries.push(tensor('scalar',p.DataType.DATA_TYPE_I32,[],packed(Int32Array.of(-7))));
    const metadata = new p.SafetensorsMetadata({entries:[
      new p.SafetensorsMetadataEntry({key:'training',value:'authoring'}),
      new p.SafetensorsMetadataEntry({key:'\u0000',value:'\u0000한글😀'}),
    ]});
    const artifact=check(await write({edits:entries.map(set),metadata}));
    const independent=unpack(artifact.data);
    assert.equal(independent.metadata['\u0000'],'\u0000한글😀');
    assert.deepEqual(independent.tensors.map(t=>t.name),entries.map(t=>t.name));
    const content=check(await read({source:artifact.data}));
    assert.equal(content.tensors.length,entries.length);
    for(let i=0;i<entries.length;i++) {
      assert.deepEqual(content.tensors[i].inline,entries[i].inline);
      assert.deepEqual(independent.tensors[i].bytes,entries[i].inline);
      assert.equal(content.tensors[i].dtype,entries[i].dtype);
      assert.deepEqual(content.tensors[i].shape,entries[i].shape); storageCases++;
    }
    const modified=check(await write({source:artifact.data,edits:[
      set(tensor('scalar',F32,[2],packed(Float32Array.of(2,4)))),
      new p.SafetensorsEdit({removeTensor:'dtype_1'}),new p.SafetensorsEdit({removeTensor:'absent'}),
      set(tensor('zeros',F32,[2,3])),
    ],removeMetadata:new p.Empty()}));
    const selected=check(await read({source:modified.data,names:['zeros','scalar']}));
    assert.deepEqual(selected.tensors[0].inline,new Uint8Array(24));
    assert.deepEqual(selected.tensors[1].inline,packed(Float32Array.of(2,4)));
    assert.equal(selected.metadata,undefined); assert.equal(unpack(modified.data).metadata,undefined);
    assert.deepEqual(check(await read({source:artifact.data,names:['scalar']})).tensors[0].inline,packed(Int32Array.of(-7))); storageCases++;
    const empty=check(await write({})); assert.equal(check(await read({source:empty.data})).tensors.length,0); storageCases++;
    assert.deepEqual(check(await write({source:artifact.data})).data,artifact.data);

    // Every half bit pattern, including signed zero, subnormals, infinities
    // and NaNs, is checked against a scalar mathematical conversion.
    const halves=Uint16Array.from({length:65536},(_,i)=>i);
    const halfFile=check(await write({edits:[set(tensor('half',p.DataType.DATA_TYPE_F16,[65536],packed(halves)))]}));
    const normalized=check(await read({source:halfFile.data,normalizeF16:true})).tensors[0];
    assert.equal(normalized.dtype,F32);
    const floats=new Float32Array(normalized.inline.buffer,normalized.inline.byteOffset,65536);
    for(let bits=0;bits<65536;bits++) {
      const sign=bits&32768?-1:1, exponent=(bits>>>10)&31, fraction=bits&1023;
      const expected=exponent===31?(fraction?NaN:sign*Infinity):exponent?sign*(1+fraction/1024)*2**(exponent-15):sign*fraction*2**-24;
      assert.ok(Object.is(floats[bits],expected),`F16 ${bits}`);
    }
    storageCases++;
    for(const request of [
      {source:Uint8Array.of(0,1)},
      {edits:[set(tensor('bad',F32,[2],new Uint8Array(4)))]},
      {edits:[set(tensor('bad',p.DataType.DATA_TYPE_F4,[1],Uint8Array.of(0)))]},
      {edits:[set(tensor('bad',F32,[-1],new Uint8Array()))]},
      {edits:[set(tensor('__metadata__',F32,[1],new Uint8Array(4)))]},
      {metadata:new p.SafetensorsMetadata({entries:[new p.SafetensorsMetadataEntry({key:'x'}),new p.SafetensorsMetadataEntry({key:'x'})]})},
    ]) {const got=await write(request);assert.notEqual(got.report.status,0);assert.equal(got.data?.length??0,0);refusals++;}
    for(const names of [['scalar','missing'],['scalar','scalar']]) {
      const got=await read({source:artifact.data,names});assert.notEqual(got.report.status,0);assert.equal(got.tensors.length,0);refusals++;
    }

    if(api.VxTrainingServiceClient) {
      assert.equal(typeof legacyInitialize,'function');
      const training=new api.VxTrainingServiceClient(reportTransport(host));
      const distributions=[['zeros','zeros',{}],['ones','ones',{}],['normal','normal',{}],
        ['normal','normal',{mean:.25,stddev:.3}],['xavierUniform','xavierUniform',{}],
        ['xavierUniform','xavierUniform',{gain:1.3}],['xavierNormal','xavierNormal',{}],['xavierNormal','xavierNormal',{gain:.8}]];
      for(const shape of [[7],[3,5],[2,3,4],[2,1,2,3]]) for(const seed of [0,1,4294967295,12345,'seed','한글','😀','\u0000',''])
        for(const [kind,field,options] of distributions) {
          const initializer=new p.TensorInitializer({
            [field]:kind==='normal'?new p.NormalInitializer(options):kind.startsWith('xavier')?new p.XavierInitializer(options):new p.Empty(),
            ...(typeof seed==='string'?{seedText:seed}:{seed}),
          });
          const result=check(await training.initializeTensor(new p.InitializeTensorRequest({
            tensor:new p.PlanningWeight({name:'initialized',dtype:F32,shape:shape.map(BigInt)}),initializer,
          })));
          const got=new Float32Array(result.tensor.inline.buffer,result.tensor.inline.byteOffset);
          const expected=legacyInitialize(shape,'float32',{type:kind,seed,...options});
          assert.equal(got.length,expected.length);
          for(let i=0;i<got.length;i++) {
            // Different native libm implementations may round the final F32
            // once differently; no RNG or distribution drift is tolerated.
            assert.ok(Math.abs(got[i]-expected[i])<=1.2e-7*Math.max(1e-20,Math.abs(expected[i])),`${kind}/${shape}/${JSON.stringify(seed)}[${i}] ${got[i]} != ${expected[i]}`);
            if(!Object.is(got[i],expected[i]))initializerRoundedDifferences++;
            initializerValues++;
          }
          initializerCases++;
        }
      for(const request of [{}, {tensor:new p.PlanningWeight({dtype:F32,shape:[]})},
        {tensor:new p.PlanningWeight({dtype:F32,shape:[2n]}),initializer:new p.TensorInitializer({normal:new p.NormalInitializer({stddev:-1})})},
        {tensor:new p.PlanningWeight({dtype:p.DataType.DATA_TYPE_I32,shape:[2n]})},
      ]) { assert.notEqual((await training.initializeTensor(new p.InitializeTensorRequest(request))).report.status,0); refusals++; }
    } else assert.equal(p.TensorInitializer,undefined);
    for(const id of retained)check(await planning.releaseGraphPlan(planRef(id)));
    assert.notEqual((await planning.editGraphPlan(new p.EditGraphPlanRequest({graphPlanId:original.graphPlanId}))).report.status,0); refusals++;
    return {status:'pass',transactions,storageCases,halfConversions:65536,refusals,initializerCases,initializerValues,initializerRoundedDifferences};
  } finally {await host.close();await foreignHost.close();}
}

if(process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
  const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href);
  const legacy=args.has('--reference')?await import(pathToFileURL(path.resolve(args.get('--reference'))).href):{};
  const result=await qualifyAuthoring(api,legacy.initializeTensor,pathToFileURL(path.resolve(args.get('--wasm'))));
  console.log(JSON.stringify(result));
  if(args.has('--out'))await writeFile(args.get('--out'),`${JSON.stringify(result,null,2)}\n`);
}
