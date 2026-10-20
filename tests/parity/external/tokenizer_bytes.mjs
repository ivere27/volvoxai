/** Lossless ByteLevel text operations through the generated portable dispatch. */
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {reportTransport,checkedReport} from '../../../tools/proto_report_fixture.mjs';

export async function qualifyTokenizerBytes(api, wasmUrl) {
  const p=api.pb, host=new (api.FullEngineHost??api.EngineHost)({wasmUrl});
  const text=new api.VxTextServiceClient(reportTransport(host)), handles=[];
  const create=async entries=>{
    const result=checkedReport(await text.createTokenizer(new p.CreateTokenizerRequest({
      byteLevel:new p.ByteLevelTokenizerVocabulary({tokens:entries})})));
    handles.push(result.tokenizerId);
    assert.equal(result.profile,p.TokenizerProfile.TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY);
    assert.equal(result.supportsEncode,false);assert.equal(result.supportsDecodeTokenBytes,true);
    assert.equal(result.tokenByteTableComplete,true);assert.equal(result.mergeCount,0);
    return result;
  };
  const bytes=(handle,tokens,skipSpecialTokens=false)=>text.decodeTokenBytes(new p.DecodeTokenBytesRequest({
    tokenizerId:handle.tokenizerId,tokens,skipSpecialTokens}));
  const decode=(handle,tokens,skipSpecialTokens=false)=>text.decodeTokens(new p.DecodeTokensRequest({
    tokenizerId:handle.tokenizerId,tokens,skipSpecialTokens}));
  let rawChecks=0,textChecks=0,refusals=0;
  try {
    const visible=[...Array.from({length:94},(_,i)=>i+33),...Array.from({length:12},(_,i)=>i+161),...Array.from({length:82},(_,i)=>i+174)];
    const alphabet=new Map(visible.map(b=>[b,String.fromCodePoint(b)]));let extra=256;
    for(let b=0;b<256;b++)if(!alphabet.has(b))alphabet.set(b,String.fromCodePoint(extra++));
    const entries=Array.from({length:256},(_,b)=>new p.ByteLevelVocabularyToken({id:b*7+1,token:alphabet.get(b)}));
    const all=await create(entries);assert.equal(all.byteAlphabetComplete,true);
    for(const entry of entries)entry.token='caller changed after publication';
    const ids=Array.from({length:256},(_,b)=>b*7+1);
    const result=checkedReport(await bytes(all,ids));assert.equal(result.complete,true);
    assert.deepEqual([...result.value],Array.from({length:256},(_,b)=>b));
    assert.deepEqual(result.tokens.map(x=>[x.tokenId,x.byteOffset,x.byteLength]),ids.map((id,i)=>[id,BigInt(i),1n]));rawChecks++;
    const utf8=new TextDecoder('utf-8',{ignoreBOM:true});
    assert.equal(checkedReport(await decode(all,ids)).text,utf8.decode(result.value));textChecks++;
    const mixed=await create([new p.ByteLevelVocabularyToken({id:7,token:'A'}),
      new p.ByteLevelVocabularyToken({id:9,token:'<eos>',added:true,special:true}),
      new p.ByteLevelVocabularyToken({id:11,token:'<think>한',added:true}),new p.ByteLevelVocabularyToken({id:13,token:''})]);
    assert.equal(mixed.byteAlphabetComplete,false);
    const skip=checkedReport(await bytes(mixed,[7,9,11,13,9,7],true));
    assert.equal(utf8.decode(skip.value),'A<think>한A');
    assert.deepEqual(skip.tokens.map(x=>[x.byteOffset,x.byteLength,x.skipped]),
      [[0n,1n,false],[1n,0n,true],[1n,10n,false],[11n,0n,false],[11n,0n,true],[11n,1n,false]]);rawChecks++;
    assert.equal(checkedReport(await decode(mixed,[7,9,11,13,9,7],true)).text,'A<think>한A');textChecks++;
    assert.equal(checkedReport(await decode(mixed,[9],false)).text,'<eos>');textChecks++;
    for(const skipSpecialTokens of [false,true])for(const method of [bytes,decode]) {
      const failure=await method(mixed,[7,4294967295],skipSpecialTokens);
      assert.equal(failure.report.code,p.OperationCode.OPERATION_CODE_TOKEN_ID_UNKNOWN);
      if('complete'in failure){
        assert.equal(failure.complete,false);
        // Empty protobuf bytes are omitted and decode as undefined. A present
        // bytes field must still be an empty Uint8Array on atomic refusal.
        if(failure.value!==undefined){assert.ok(failure.value instanceof Uint8Array);assert.equal(failure.value.length,0);}
        assert.deepEqual(failure.tokens,[]);
      }
      else assert.equal(failure.text,'');refusals++;
    }
    const unsupported=await text.encodeText(new p.EncodeTextRequest({tokenizerId:mixed.tokenizerId,text:'A'}));
    assert.equal(unsupported.report.code,p.OperationCode.OPERATION_CODE_TOKENIZER_OPERATION_UNSUPPORTED);assert.deepEqual(unsupported.tokens,[]);refusals++;
    // BOM and invalid bytes must survive BytesRPC. Once-message text decoding
    // joins split multibyte sequences and preserves the ByteLevel initial BOM.
    const raw=[0xef,0xbb,0xbf,0xe2,0x82,0xac,0xff,0];
    const bomIds=raw.map(b=>b*7+1),bom=checkedReport(await bytes(all,bomIds));
    assert.deepEqual([...bom.value],raw);rawChecks++;
    assert.equal(checkedReport(await decode(all,bomIds)).text,'\ufeff€�\0');textChecks++;
    const stream=new TextDecoder('utf-8',{ignoreBOM:true});let streamed='';
    for(const id of bomIds)streamed+=stream.decode(checkedReport(await bytes(all,[id])).value,{stream:true});
    streamed+=stream.decode();assert.equal(streamed,'\ufeff€�\0');textChecks++;
    for(const definition of [[new p.ByteLevelVocabularyToken({id:0,token:'한'})],
      [new p.ByteLevelVocabularyToken({id:0,token:'A',special:true})],
      [new p.ByteLevelVocabularyToken({id:0,token:'A'}),new p.ByteLevelVocabularyToken({id:0,token:'B'})]]) {
      const failure=await text.createTokenizer(new p.CreateTokenizerRequest({byteLevel:new p.ByteLevelTokenizerVocabulary({tokens:definition})}));
      assert.notEqual(failure.report.status,0);assert.equal(failure.tokenizerId,0n);refusals++;
    }
    return {status:'PASS',rawChecks,textChecks,refusals,scope:'tiny public ByteLevel RPC fixture; no HF encoding or model execution'};
  } finally {
    for(const tokenizerId of handles)checkedReport(await text.releaseTokenizer(new p.TokenizerRef({tokenizerId})));
    await host.close();
  }
}

if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
  const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
  const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href);
  const result=await qualifyTokenizerBytes(api,pathToFileURL(path.resolve(args.get('--wasm'))));
  console.log(JSON.stringify(result));if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify(result,null,2)+'\n');
}
