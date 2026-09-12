/** Right-aligned matrix broadcasting and affine views, with scalar oracles. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {graphOf,runGraph,compare,requirePhysicalDevice} from './webgpu_device_bridge.mjs';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(`file://${path.resolve(args.get('--bundle'))}`);
const {device,description}=await requirePhysicalDevice(),options={device,wasm:path.resolve(args.get('--wasm'))};
const records=[];
const product=shape=>shape.reduce((a,b)=>a*b,1);
const check=async(name,graph,batches,expected)=>{
  if(args.has('--only')&&!name.includes(args.get('--only')))return;
  for(const backend of ['wasm','webgpu']) {
    const result=await runGraph(api,options,graph,batches,backend);
    for(let round=0;round<batches.length;round++)compare(`${name}/${backend}/${round}`,result.rounds[round].get('y'),expected[round],1e-5);
    if(backend==='webgpu')assert.equal(result.route.fallbackNodes,0);
  }
  records.push({name,bindings:batches.length});console.log(`layout variants ${name}: PASS`);
};
try {
  for(const [dtype,Array_] of [['float32',Float32Array],['int8',Int8Array],['uint8',Uint8Array]]) {
    for(const [ar,br] of [[2,3],[3,2],[3,4],[4,3],[2,8],[8,2],[4,4]]) {
      const rank=Math.max(ar,br),out=[...Array(rank-2).fill(2),2,3];out[0]='B';
      const a=[...out.slice(rank-ar,-2),2,3],b=[...out.slice(rank-br,-2),3,3];
      // The equal-rank case broadcasts both operands on different batch axes.
      if(ar===br){a[1]=1;b[0]=1;}
      const quantized=dtype!=='float32',weights={},quantization={};
      if(quantized)for(const name of ['a','b','y']) {
        weights[name+'.scale']={shape:[1],data:Float32Array.of(1)};
        weights[name+'.zero']={shape:[1],data:Array_.of(0)};
        quantization[name]={scheme:'per_tensor',scale_tensor:name+'.scale',zero_point_tensor:name+'.zero'};
      }
      const graph=graphOf([{id:'n',opType:quantized?'QBatchMatMul':'BatchMatMul',inputs:{a:'a',b:'b'},
        outputs:{out:{tensor:'y',dtype,shape:out}},params:{}}],{a:{shape:a,dtype},b:{shape:b,dtype}},['y'],weights,quantized?quantization:undefined);
      graph.dimensions={B:{min:1,max:3}};
      const batches=[1,3,2,1].map(batch=>[a,b].map((shape,i)=>{
        const concrete=shape.map(d=>d==='B'?batch:d);
        return {name:i?'b':'a',shape:concrete,data:Array_.from({length:product(concrete)},(_,j)=>(j+i)%3-(dtype==='uint8'?0:1))};
      }));
      const expected=batches.map(([left,right],round)=>{
        const shape=out.map(d=>d==='B'?[1,3,2,1][round]:d),values=[];
        for(let batch=0;batch<product(shape.slice(0,-2));batch++) {
          let remaining=batch;const coordinates=Array(rank-2);
          for(let axis=rank-3;axis>=0;axis--){coordinates[axis]=remaining%shape[axis];remaining=Math.floor(remaining/shape[axis]);}
          const offset=t=>{
            let at=0;for(let axis=0;axis<t.shape.length-2;axis++)
              at=at*t.shape[axis]+(t.shape[axis]===1?0:coordinates[rank-t.shape.length+axis]);
            return at*product(t.shape.slice(-2));
          };
          const ai=offset(left),bi=offset(right);
          for(let m=0;m<2;m++)for(let n=0;n<3;n++) {
            let sum=0;for(let k=0;k<3;k++)sum+=left.data[ai+m*3+k]*right.data[bi+k*3+n];values.push(sum);
          }
        }return Array_.from(values);
      });
      await check(`${quantized?'QBatchMatMul':'BatchMatMul'}/${dtype}/${ar}x${br}`,graph,batches,expected);
    }
  }
  for(const [dtype,Array_] of [['int8',Int8Array],['uint8',Uint8Array]])for(const op of ['Identity','Reshape']) {
    const shape=[3,5],values=Array_.from({length:15},(_,i)=>i*31-129);
    const weights={scale:{shape:[5],data:Float32Array.of(.01,.02,.03,.04,.05)},zero:{shape:[5],data:Array_.of(0,1,2,3,4)}};
    const q={scheme:'per_axis',axis:1,scale_tensor:'scale',zero_point_tensor:'zero'};
    const graph=graphOf([{id:'n',opType:op,inputs:{input:'x'},outputs:{out:{tensor:'y',dtype,shape}},params:op==='Reshape'?{shape}:{}}],
      {x:{shape,dtype}},['y'],weights,{x:q,y:q});
    await check(`${op}/${dtype}/per-axis-tail`,graph,[[{name:'x',shape,data:values}]],[values]);
  }
  assert.ok(records.length);console.log(`layout variants: ${records.length} cases passed`);
  if(args.has('--out'))await Deno.writeTextFile(args.get('--out'),JSON.stringify({adapter:description,records},null,2));
}finally{device.destroy();await device.lost;}
