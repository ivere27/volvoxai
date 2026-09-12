/** Shared graphs for inference and independent-gradient qualification. */
const product = shape => shape.reduce((a,b)=>a*b,1);
const tensor = (shape, seed=0) => ({shape, values:Float32Array.from({length:product(shape)},(_,i)=>Math.sin(i+seed+.3)*.35)});
const node = (id,opType,inputs,shape,params={}) => ({id,opType,inputs,
  outputs:{out:{tensor:id,dtype:'float32',shape}},params});
const definitions = [];
function operatorFixture(name,shape,outputShape,op,params={},extras={},ports={}) {
  const width=shape.at(-1),outputWidth=outputShape.at(-1);
  const weights={projection:tensor([width,width],11),...extras,head:tensor([outputWidth,2],12)};
  definitions.push({name,shape,weights,rows:product(outputShape)/outputWidth,nodes:[
    node('projected','MatMul',{input:'x',weight:'projection'},shape,{weight_layout:'din_dout'}),
    node('transformed',op,{...(!('a' in ports || 'q' in ports || 'x' in ports)?{input:'projected'}:{}),...ports},outputShape,params),
    node('logits','MatMul',{input:'transformed',weight:'head'},[...outputShape.slice(0,-1),2],{weight_layout:'din_dout'}),
  ]});
}
operatorFixture('Clip-tensor-bounds',[2,3],[2,3],'Clip',{},
  {lower:{shape:[1],values:Float32Array.of(-.12)},upper:{shape:[1],values:Float32Array.of(.14)}},
  {min:'lower',max:'upper'});
operatorFixture('Conv2D-no-bias',[1,3,4,2],[1,2,3,3],'Conv2D',{weight_layout:'HWIO'},
  {conv:tensor([2,2,2,3],3)},{weight:'conv'});
for(const sigmoid of [false,true])operatorFixture('Concat2'+(sigmoid?'-sigmoid':''),[2,3],[2,5],'Concat2',{axis:1,sigmoid},
  {right:tensor([2,2],3)},{b:'right',a:'projected'});
operatorFixture('Conv1D-grouped',[2,5,4],[2,5,6],'Conv1D',{groups:2,stride:1,padding:1,relu:1},
  {conv:tensor([3,2,6],2),bias:tensor([6],3)},{weight:'conv',bias:'bias'});
operatorFixture('ConvTranspose2D',[1,2,3,2],[1,4,5,4],'ConvTranspose2D',{stride:[2,2],padding:[0,1],kernel:[2,3]},
  {conv:tensor([2,3,2,4],2),bias:tensor([4],3)},{weight:'conv',bias:'bias'});
operatorFixture('Pad',[1,2,3,2],[1,4,5,2],'Pad',{pads:[0,1,2,0,0,1,0,0],value:.2});
for(const mode of ['nearest','linear'])operatorFixture('Resize-'+mode,[2,2,3,2],[2,5,4,2],'Resize',{mode});
operatorFixture('Interpolate1D',[2,3,4],[2,3,7],'Interpolate1D',{size:7});
for(const op of ['SpatialSoftargmaxY','MeanHeight','ProfileX','ProfileY'])
  operatorFixture(op,[2,3,4,2],[2,op.startsWith('Profile')?4:2,op==='ProfileY'?3:4],op);
operatorFixture('Expand',[2,1,3],[2,4,3],'Expand',{shape:[2,4,3]});
operatorFixture('Slice',[2,4],[2,2],'Slice',{axes:[1],starts:[1],ends:[4],steps:[2]});
operatorFixture('Gather-repeated',[2,4],[2,3],'Gather',{axis:1},
  {indices:{shape:[3],values:new Int32Array([2,2,0]),dtype:'I32'}},{indices:'indices'});
operatorFixture('GatherElements-negative-repeated',[2,4],[2,3],'GatherElements',{axis:1},
  {indices:{shape:[2,3],values:new Int32Array([-1,1,1,0,0,2]),dtype:'I32'}},{indices:'indices'});
for(const op of ['Add','Mul','Sub','Div'])operatorFixture(op,[2,3],[2,3],op,{},
  {right:{shape:[3],values:new Float32Array([.7,.8,.9])}},{a:'projected',b:'right'});
for(const relu of [1,2])operatorFixture('Add-broadcast-relu'+relu,[2,3],[2,3],'Add',{relu},
  {right:{shape:[3],values:Float32Array.of(-.1,.2,6.1)}},{a:'projected',b:'right'});
for(const op of ['Add','Mul'])operatorFixture(op+'-both-broadcast',[2,1,3],[2,4,3],op,{},
  {right:tensor([1,4,1],2)},{a:'projected',b:'right'});
for(const op of ['Add','Mul','Sub','Div']) definitions.push({name:op+'-rank-zero',shape:[],rows:1,
  weights:{projection:{shape:[],values:Float32Array.of(.7)},right:{shape:[],values:Float32Array.of(1.3)},head:tensor([1,2],12)},
  nodes:[node('projected','Mul',{a:'x',b:'projection'},[]),node('scalar',op,{a:'projected',b:'right'},[]),
    node('view','Reshape',{input:'scalar'},[1,1],{shape:[1,1]}),
    node('logits','MatMul',{input:'view',weight:'head'},[1,2],{weight_layout:'din_dout'})]});
operatorFixture('Where',[2,3],[2,3],'Where',{},
  {condition:{shape:[2,3],values:new Int32Array([1,0,0,1,1,0]),dtype:'I32'},right:tensor([2,3],4)},
  {condition:'condition',a:'projected',b:'right'});
operatorFixture('DequantizeLinear-F32',[2,3],[2,3],'DequantizeLinear',{},
  {scale:{shape:[1],values:new Float32Array([.7])},zero:{shape:[1],values:new Int32Array([1]),dtype:'I32'}},
  {scale:'scale',zero_point:'zero'});
operatorFixture('CrossAttention',[2,3,4],[2,3,4],'CrossAttention',{heads:2},
  {kv:tensor([2,2,4],2),attention:tensor([12,4],3),scale:tensor([12],4),bias:tensor([12],5)},
  {q:'projected',kv:'kv',weight:'attention',scale:'scale',bias:'bias'});
for (const op of ['SDPA','CrossSDPA']) for (const dropout of [0,.25]) {
  const shape=[2,3,4], weights={projection:tensor([4,op==='SDPA'?12:4],1),head:tensor([4,2],2)};
  const nodes=[node('projected','MatMul',{input:'x',weight:'projection'},[2,3,op==='SDPA'?12:4],{weight_layout:'din_dout'})];
  const params={heads:2,causal:true,attention_dropout:dropout,dropout_seed:19};
  if(op==='SDPA') nodes.push(node('attended',op,{qkv:'projected'},shape,params));
  else {
    weights.keys=tensor(shape,3); weights.values=tensor(shape,4);
    nodes.push(node('attended',op,{q:'projected',k:'keys',v:'values'},shape,params));
  }
  nodes.push(node('logits','MatMul',{input:'attended',weight:'head'},[2,3,2],{weight_layout:'din_dout'}));
  definitions.push({name:op+(dropout?'-dropout':''),shape,weights,nodes,rows:6});
}
{
  const weights={projection:tensor([3,4],1),router:tensor([4,3],2),bias:tensor([3],3),experts:tensor([3,4,2],4),expert_bias:tensor([3,2],5)};
  definitions.push({name:'MoERouter-MoELinear',shape:[2,3],weights,rows:2,nodes:[
    node('projected','MatMul',{input:'x',weight:'projection'},[2,4],{weight_layout:'din_dout'}),
    {id:'router',opType:'MoERouter',inputs:{input:'projected',weight:'router',bias:'bias'},
      outputs:{indices:{tensor:'indices',dtype:'float32',shape:[2,2]},weights:{tensor:'gates',dtype:'float32',shape:[2,2]}},
      params:{num_experts:3,top_k:2,normalize:true,temperature:.8}},
    node('logits','MoELinear',{input:'projected',expert_weight:'experts',expert_bias:'expert_bias',route_indices:'indices',route_weights:'gates'},[2,2]),
  ]});
}

// One wide fan-out/fan-in graph checks both port storage and gradient
// accumulation from repeated operands. Only two split outputs feed the loss.
for(const count of [65,257]) {
  const shape=[2,2],ports={},outputs={};
  for(let i=count-1;i>=0;i--) ports['input'+i]='projected';
  for(let i=0;i<count;i++) outputs['out'+i]={tensor:'piece'+i,dtype:'float32',shape};
  definitions.push({name:'Concat-Split-'+count,shape,rows:2,
    weights:{projection:tensor([2,2],1),head:tensor([2,2],2)},nodes:[
      node('projected','MatMul',{input:'x',weight:'projection'},shape,{weight_layout:'din_dout'}),
      node('joined','Concat',ports,[2,count*2],{axis:1}),
      {id:'split',opType:'Split',inputs:{input:'joined'},outputs,params:{axis:1,split:Array(count).fill(2)}},
      node('summed','Add',{a:'piece0',b:'piece'+(count-1)},shape),
      node('logits','MatMul',{input:'summed',weight:'head'},[2,2],{weight_layout:'din_dout'}),
    ]});
}

// Previously existing adjoints need the same independent numerical derivative
// check as newly migrated operators. Inputs stay away from discontinuities.
for (const op of ['ReLU', 'Sigmoid', 'Tanh', 'SiLU', 'GELU', 'LeakyReLU', 'HardSigmoid', 'HardSwish'])
  operatorFixture(op, [2,3], [2,3], op, op === 'LeakyReLU' ? {alpha:.17} : {});
operatorFixture('GELU-tanh',[2,3],[2,3],'GELU',{approximate:'tanh'});
for(const op of ['Softmax','LogSoftmax']) operatorFixture(op,[2,3],[2,3],op,{axis:-1});
for(const op of ['LayerNorm','RMSNorm']) operatorFixture(op,[2,3],[2,3],op,{eps:1e-4},
  {gamma:tensor([3],2),...(op==='LayerNorm'?{beta:tensor([3],3)}:{})},
  {weight:'gamma',...(op==='LayerNorm'?{bias:'beta'}:{})});
operatorFixture('PReLU',[2,3],[2,3],'PReLU',{}, {slope:tensor([3],2)}, {slope:'slope'});
operatorFixture('GroupNorm',[2,2,3,4],[2,2,3,4],'GroupNorm',{num_groups:2,eps:1e-4},
  {gamma:tensor([4],2),beta:tensor([4],3)}, {weight:'gamma',bias:'beta'});
operatorFixture('BatchNorm2D',[2,2,3,4],[2,2,3,4],'BatchNorm2D',{eps:1e-4},
  {gamma:tensor([4],2),beta:tensor([4],3),running_mean:tensor([4],4),running_var:{shape:[4],values:Float32Array.of(.8,1,1.2,1.4)}},
  {weight:'gamma',bias:'beta',running_mean:'running_mean',running_var:'running_var'});
operatorFixture('MaxPool2D',[2,4,4,2],[2,2,2,2],'MaxPool2D',{kernel:[2,2],stride:[2,2]});
operatorFixture('GlobalAveragePool',[2,2,3,4],[2,1,1,4],'GlobalAveragePool');
for(const op of ['ReduceSum','ReduceMean']) operatorFixture(op,[2,3,4],[2,3],op,{axis:-1,keepdims:false});
operatorFixture('Transpose',[2,3,4],[2,4,3],'Transpose',{perm:[0,2,1]});
operatorFixture('Identity',[2,3],[2,3],'Identity');
operatorFixture('Flatten',[2,2,3],[2,6],'Flatten',{axis:1});
operatorFixture('Squeeze',[2,1,3],[2,3],'Squeeze',{axes:[1]});
operatorFixture('Unsqueeze',[2,3],[2,1,3],'Unsqueeze',{axes:[1]});
operatorFixture('Reshape',[2,2,3],[2,6],'Reshape',{shape:[2,6]});
operatorFixture('Cast-F32',[2,3],[2,3],'Cast',{to:'float32'});
for(const probability of [0,.25]) operatorFixture('Dropout-'+probability,[2,3],[2,3],'Dropout',{p:probability,seed:19});
operatorFixture('Embedding-repeated',[3,3],[3,3],'Embedding',{},
  {table:tensor([4,3],2),ids:{shape:[3],dtype:'I32',values:Int32Array.of(1,1,2)}}, {input:'ids',weight:'table'});
// Both trainable branches must reach the loss: requesting a disconnected
// trainable is rejected by the original trainer and the C implementation.
definitions.at(-1).nodes.splice(2,0,node('combined','Add',{a:'projected',b:'transformed'},[3,3]));
definitions.at(-1).nodes[3].inputs.input='combined';

function shard(weights) {
  const header={}, parts=[];let offset=0;
  for(const [name,{shape,values,dtype='F32'}] of Object.entries(weights)) {
    const bytes=new Uint8Array(values.buffer,values.byteOffset,values.byteLength);
    header[name]={dtype,shape,data_offsets:[offset,offset+bytes.length]};
    offset+=bytes.length;parts.push(bytes);
  }
  let text=JSON.stringify(header);text+=' '.repeat((8-text.length%8)%8);
  const data=new Uint8Array(8+text.length+offset);
  new DataView(data.buffer).setBigUint64(0,BigInt(text.length),true);
  data.set(new TextEncoder().encode(text),8);offset=8+text.length;
  for(const bytes of parts){data.set(bytes,offset);offset+=bytes.length;}
  return data;
}

export { definitions, tensor, shard };
