/** Portable dtype matrix, packed tails and typed views against JS storage semantics. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {graphOf,runGraph,requirePhysicalDevice} from './webgpu_device_bridge.mjs';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(`file://${path.resolve(args.get('--bundle'))}`);
const {device,description}=await requirePhysicalDevice(),options={device,wasm:path.resolve(args.get('--wasm'))};
const records=[];
const kinds=[['float32',Float32Array],['int32',Int32Array],['int8',Int8Array],['uint8',Uint8Array]];
const node=(op,shape,dtype,params={})=>({id:'n',opType:op,inputs:{input:'x'},outputs:{out:{tensor:'y',shape,dtype}},params});
const fixture=(op,inputShape,outputShape,dtype,params={})=>graphOf([node(op,outputShape,dtype,params)],{x:{shape:inputShape,dtype}},['y']);
const affine=graph=>{
  const q={scheme:'per_tensor',scale_tensor:'scale',zero_point_tensor:'zero'};
  graph.quantization={format:'volvox-affine-safetensors/v1',tensors:{x:q,y:q}};
  const Zero=graph.inputs.x.dtype==='int8'?Int8Array:Uint8Array;
  graph.__weights={scale:{shape:[1],data:Float32Array.of(.03)},zero:{shape:[1],data:Zero.of(0)}};
  return graph;
};
const check=async(name,graph,batches,expected)=>{
  if(args.has('--only')&&!name.includes(args.get('--only')))return;
  for(const backend of ['wasm','webgpu']) {
    const result=await runGraph(api,options,graph,batches,backend);
    for(let round=0;round<result.rounds.length;round++) {
      const actual=result.rounds[round].get('y');
      assert.deepEqual([...actual],[...expected[round]],`${name}/${backend}/${round}`);
    }
    if(backend==='webgpu')assert.equal(result.route.fallbackNodes,0);
  }
  records.push({name,bindings:batches.length});console.log(`typed storage ${name}: PASS`);
};
try {
  for(const [from,Source] of kinds)for(const [to,Target] of kinds) {
    const graph=fixture('Cast',['N'],['N'],to,{to});graph.inputs.x.dtype=from;
    graph.dimensions={N:{min:1,max:23}};
    // JS TypedArray conversion is an independent oracle for large finite
    // values, infinities, signed zero, truncation and modulo narrowing.
    const values=Source.from([NaN,Infinity,-Infinity,-0,-129.8,128.9,256.1,-257.5,4294967552,-4294967552,16777217,3.9,-3.9,0,1,2,3,4,5,6,7,8,9]);
    const batches=[1,23,7,5].map(n=>[{name:'x',shape:[n],data:values.slice(0,n)}]);
    await check(`Cast/${from}/${to}`,graph,batches,batches.map(batch=>Target.from(batch[0].data)));
  }
  for(const [dtype,Array_] of kinds.slice(1)) {
    for(const [op,shape,out,params] of [
      ['Identity',[3,5],[3,5],{}],['Flatten',[3,1,5],[3,5],{axis:1}],
      ['Reshape',[3,5],[5,3],{shape:[5,3]}],['Squeeze',[3,1,5],[3,5],{axes:[1]}],
      ['Unsqueeze',[3,5],[3,1,5],{axes:[1]}]
    ]) {
      const graph=fixture(op,shape,out,dtype,params);if(dtype!=='int32')affine(graph);
      const values=Array_.from({length:15},(_,i)=>i*37-129),next=Array_.from(values,v=>v+3);
      await check(`${op}/${dtype}/tail`,graph,[[{name:'x',shape,data:values}],[{name:'x',shape,data:next}]],[values,next]);
    }
  }
  for(const [dtype,Array_] of kinds.slice(2)) {
    {
      const graph=affine(fixture('Expand',[1,5],[3,5],dtype,{shape:[3,5]}));
      const values=Array_.of(-120,-1,0,5,127),expected=Array_.from([...values,...values,...values]);
      await check(`Expand/${dtype}/tail`,graph,[[{name:'x',shape:[1,5],data:values}]],[expected]);
    }
    for(const op of ['MaxPool2D','Resize']) {
      const shape=['B',3,5,1],out=op==='MaxPool2D'?['B',2,3,1]:['B',5,7,1];
      const graph=affine(fixture(op,shape,out,dtype,op==='MaxPool2D'
        ?{kernel:[2,2],stride:[2,2],pads:[1,1,0,0]}:{mode:'nearest'}));
      graph.dimensions={B:{min:1,max:3}};
      const batches=[1,3,2,1].map(b=>[{name:'x',shape:[b,3,5,1],data:Array_.from({length:b*15},(_,i)=>i*37-129)}]);
      const expected=batches.map(([input])=>{
        const values=[];for(let b=0;b<input.shape[0];b++)for(let y=0;y<out[1];y++)for(let x=0;x<out[2];x++) {
          if(op==='Resize')values.push(input.data[b*15+Math.floor(y*3/5)*5+Math.floor(x*5/7)]);
          else {
            let maximum=dtype==='int8'?-128:0;
            for(let yy=0;yy<2;yy++)for(let xx=0;xx<2;xx++) {
              const iy=y*2+yy-1,ix=x*2+xx-1;
              if(iy>=0&&iy<3&&ix>=0&&ix<5)maximum=Math.max(maximum,input.data[b*15+iy*5+ix]);
            }values.push(maximum);
          }
        }return Array_.from(values);
      });
      await check(`${op}/${dtype}/dynamic-tail`,graph,batches,expected);
    }
  }
  assert.ok(records.length);console.log(`typed storage: ${records.length} cases passed`);
  if(args.has('--out'))await Deno.writeTextFile(args.get('--out'),JSON.stringify({adapter:description,records},null,2));
}finally{device.destroy();await device.lost;}
