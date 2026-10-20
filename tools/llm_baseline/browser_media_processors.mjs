/** Development caller processors. Nothing in this file is a product entry. */
export function whisperFeatures(pcm, processor) {
  if (!(pcm instanceof Float32Array) || !pcm.length || pcm.some(v => !Number.isFinite(v))) throw Error('finite mono F32 PCM required');
  const expected={sampling_rate:16000,n_fft:400,hop_length:160,n_samples:480000,feature_size:80,nb_max_frames:3000,padding_side:'right',padding_value:0,chunk_length:30};
  for(const [name,value]of Object.entries(expected))if(processor[name]!==value)throw Error('unsupported processor '+name);
  const filters=processor.mel_filters;
  if(filters.length!==80||filters.some(row=>row.length!==201||row.some(v=>!Number.isFinite(v)||v<0)))throw Error('pinned Slaney filters required');
  const waveform=new Float64Array(480000);waveform.set(pcm.subarray(0,480000));
  const window=Float64Array.from({length:400},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/400));
  function plan(n,offset=0,stride=1) {
    const radix=n%2===0?2:5,m=n/radix;
    const children=n===1?[]:Array.from({length:radix},(_,i)=>plan(m,offset+i*stride,stride*radix));
    const angles=n===1?[]:Array.from({length:n},(_,i)=>[Math.cos(-2*Math.PI*i/n),Math.sin(-2*Math.PI*i/n)]);
    const real=new Float64Array(n),imag=new Float64Array(n);
    return {real,imag,run(input){
      if(n===1){real[0]=input[offset];imag[0]=0;return;}
      for(const child of children)child.run(input);
      for(let k=0;k<n;k++){
        let re=0,im=0;
        for(let j=0;j<radix;j++){
          const child=children[j],i=k%m,[c,s]=angles[(j*k)%n];
          re+=child.real[i]*c-child.imag[i]*s;im+=child.real[i]*s+child.imag[i]*c;
        }
        real[k]=re;imag[k]=im;
      }
    }};
  }
  const fft=plan(400),frame=new Float64Array(400),power=new Float64Array(201),result=new Float32Array(80*3000);
  let maximum=-Infinity;
  for(let t=0;t<3000;t++){
    for(let i=0;i<400;i++){
      let index=t*160+i-200;
      if(index<0)index=-index;else if(index>=480000)index=959998-index;
      frame[i]=waveform[index]*window[i];
    }
    fft.run(frame);
    for(let i=0;i<201;i++){const re=Math.fround(fft.real[i]),im=Math.fround(fft.imag[i]);power[i]=re*re+im*im;}
    for(let mel=0;mel<80;mel++){
      let value=0;for(let i=0;i<201;i++)value+=filters[mel][i]*power[i];
      const rounded=Math.fround(Math.log10(Math.max(value,1e-10)));result[mel*3000+t]=rounded;maximum=Math.max(maximum,rounded);
    }
  }
  const floor=Math.fround(maximum-8);
  for(let i=0;i<result.length;i++)result[i]=Math.fround(Math.fround(Math.max(result[i],floor)+4)/4);
  return {values:result,shape:[1,80,3000],dtype:'F32'};
}

export async function receiptPixels(jpeg) {
  const bitmap=await createImageBitmap(new Blob([jpeg],{type:'image/jpeg'}),{colorSpaceConversion:'none'});
  // The qualified JPEGs already have the model's exact geometry. Different
  // sizes require a separately qualified resize branch, never a silent resize.
  if(bitmap.width!==672||bitmap.height!==320){bitmap.close();throw Error('qualified JPEG geometry is 672 x 320');}
  const canvas=new OffscreenCanvas(672,320),context=canvas.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});
  context.drawImage(bitmap,0,0);bitmap.close();
  const rgba=context.getImageData(0,0,672,320).data,values=new Float32Array(672*320);
  for(let i=0;i<values.length;i++){
    // Pillow's RGB -> L integer coefficients and F32 normalization boundaries.
    const gray=(19595*rgba[4*i]+38470*rgba[4*i+1]+7471*rgba[4*i+2]+32768)>>16;
    values[i]=Math.fround(Math.fround(Math.fround(gray/255)-.5)/.5);
  }
  return {values,shape:[1,1,320,672],dtype:'F32',decoder:'Chrome createImageBitmap colorSpaceConversion=none; exact-size; Pillow RGB-to-L coefficients'};
}

export function receiptQuestion(text,vocabulary) {
  const stoi=new Map(vocabulary.itos.map((value,index)=>[value,index])),unused=new Set(vocabulary.unused_tokens),byteSet=new Set(vocabulary.byte_tokens);
  const tags=vocabulary.atomic_tokens.filter(v=>v.length>1).sort((a,b)=>b.length-a.length||a.localeCompare(b));
  const ranks=new Map(vocabulary.merges.map((pair,index)=>[JSON.stringify(pair),index]));
  const fallback=character=>Array.from(new TextEncoder().encode(character),v=>`<0x${v.toString(16).toUpperCase().padStart(2,'0')}>`);
  const mergeSpan=span=>{
    let tokens=Array.from(span).flatMap(c=>stoi.has(c)&&!byteSet.has(c)&&!unused.has(c)?[c]:fallback(c));
    while(tokens.length>1){
      let best,rank=Infinity;
      for(let i=0;i<tokens.length-1;i++){const pair=JSON.stringify(tokens.slice(i,i+2)),r=ranks.get(pair);if(r!==undefined&&r<rank){rank=r;best=pair;}}
      if(best===undefined)break;
      const merged=[];for(let i=0;i<tokens.length;){if(i+1<tokens.length&&JSON.stringify(tokens.slice(i,i+2))===best){merged.push(tokens[i]+tokens[i+1]);i+=2;}else merged.push(tokens[i++]);}tokens=merged;
    }
    return tokens.map(t=>{if(!stoi.has(t))throw Error('unregistered caller token '+t);return stoi.get(t);});
  };
  const normalized=String(text).replace(/\s+/gu,' ').trim().normalize('NFC'),ids=[];
  let start=0,cursor=0;
  while(cursor<normalized.length){
    const tag=tags.find(t=>normalized.startsWith(t,cursor)),c=String.fromCodePoint(normalized.codePointAt(cursor));
    if(!tag&&!/[0-9\s]/u.test(c)){cursor+=c.length;continue;}
    if(start<cursor)ids.push(...mergeSpan(normalized.slice(start,cursor)));
    if(tag){ids.push(stoi.get(tag));cursor+=tag.length;}else{ids.push(...(stoi.has(c)&&!byteSet.has(c)?[stoi.get(c)]:fallback(c).map(t=>stoi.get(t))));cursor+=c.length;}start=cursor;
  }
  if(start<cursor)ids.push(...mergeSpan(normalized.slice(start)));
  ids.push(2);if(ids.length>192){ids.length=192;ids[191]=2;}return ids;
}
