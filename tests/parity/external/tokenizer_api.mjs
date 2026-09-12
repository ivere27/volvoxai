import { reportTransport, checkedReport } from '../../../tools/proto_report_fixture.mjs';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const encoder = new TextEncoder();
function binary(tokens) {
  const bytes = new Uint8Array(4 + tokens.reduce((sum, token) => sum + 4 + token.length, 0));
  const view = new DataView(bytes.buffer); view.setInt32(0, tokens.length, true);
  let at = 4;
  for (const token of tokens) { view.setInt32(at, token.length, true); at += 4; bytes.set(token, at); at += token.length; }
  return bytes;
}

export async function qualifyTokenizer(api, Legacy, wasmUrl) {
  const p = api.pb, host = new (api.FullEngineHost ?? api.EngineHost)({ wasmUrl });
  const host2 = new (api.FullEngineHost ?? api.EngineHost)({ wasmUrl });
  const service = new api.VxTextServiceClient(reportTransport(host)), foreign = new api.VxTextServiceClient(reportTransport(host2));
  let comparisons = 0, refusals = 0;
  const check = checkedReport;
  const seedWords = ['ab','abc','bc','aa',' a',' ab',' abc','한','글','한글',' 한',' 한글','é','éé',' e','  ','\t\t','12','123',' 1',' 12',"'s","'t","'re","'ve","'m","'ll","'d",'!!',' !!','😀','😀😀','\u0301','\ufeff','Ġ','Ġa','𝒜','𝒜𝒜','Ⅳ','ⅣⅣ','𠀀','𠀀𠀀'];
  const raw = [...Array.from({length:256},(_,i)=>Uint8Array.of(i)), ...seedWords.map(word=>encoder.encode(word)), Uint8Array.of(0xe2,0x82), Uint8Array.of(0xed,0xa0,0x80), Uint8Array.of(0xff), new Uint8Array()];
  const mergeText = '#version: 0.2\r\na b\nb c\nab c\na a\nĠ a\nĠa b\nĠab c\n한 글\nĠ 한\nĠ한 글\né é\n1 2\n12 3\nĠ 1\nĠ1 2\n! !\nĠ !\nĠ! !\n😀 😀\n𝒜 𝒜\nⅣ Ⅳ\n𠀀 𠀀\nmissing pair\na b ignored\n';
  const corpus = [ '', 'abc bc aa ab', '  abc\t\nabc  ', "we're abc's I'd I'll abc't've'd", '한글 한글', 'éé e\u0301', '😀😀 ! 😀', '123 123 abc123', '\u0000abc\u0000', '\ufeffabc\ufeff', 'ⅣⅣ 𝒜𝒜 𠀀𠀀', '\u0085abc\u200babc\u00a0abc\u2028abc\u3000abc', 'abc'.repeat(200) ];
  const alphabet = Array.from("abc123 !\t\r\n'한글😀é\u0301\u00a0\u2009\u200bⅣ𝒜𠀀Ġ\u0000");
  let seed = 18273;
  for(let i=0;i<120;i++) {let text='';for(let j=0;j<1+i%40;j++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;text+=alphabet[seed%alphabet.length];}corpus.push(text);}
  try {
    for (const format of ['binary','tokens','json']) for(const bpe of [false,true]) {
      const legacy = new Legacy();
      const jsonWords = [...new Set(['',...Array.from({length:128},(_,i)=>String.fromCharCode(i)), ...seedWords])];
      const json = Object.fromEntries(jsonWords.map((word,i)=>[word,i*3]));
      const vocab = format==='json' ? encoder.encode(JSON.stringify(json)) : binary(raw);
      const request = new p.CreateTokenizerRequest({
        ...(format==='json' ? {vocabularyJson:vocab} : format==='binary' ? {vocabularyBinary:vocab} : {tokens:new p.TokenizerVocabulary({tokens:raw.map((value,id)=>new p.VocabularyToken({id,value}))})}),
        merges:encoder.encode(bpe?mergeText:''),
      });
      // The oracle is the unchanged TypeScript file from 99dfd8f; its original
      // loaders parse the same bytes. Fetch is replaced only in this test scope.
      const originalFetch=globalThis.fetch, originalLog=console.log;
      globalThis.fetch=async url=>new Response(String(url).endsWith('merges.txt')?mergeText:vocab);
      console.log=()=>{};
      try {await legacy.load(format==='json'?'vocab.json':'vocab.bin',bpe?'merges.txt':null);}
      finally {globalThis.fetch=originalFetch;console.log=originalLog;}
      const handle=check(await service.createTokenizer(request));
      const id=handle.tokenizerId;
      assert.equal(handle.vocabularySize,format==='json'?jsonWords.length:raw.length);
      assert.equal(handle.mergeCount,legacy.merges.size);
      const invalid=await foreign.encodeText(new p.EncodeTextRequest({tokenizerId:id,text:'abc'}));
      assert.notEqual(invalid.report.status,0);refusals++;
      // Creation snapshots the request, and encoding has no mutable cursor.
      request.merges.fill(0); if(request.vocabularyBinary)request.vocabularyBinary.fill(0);
      for(const text of corpus) for(const mode of [0,1,2]) {
        const limit=[0,1,4,256,2048][comparisons%5];
        const expected=mode===1?legacy.encodeGreedy(text,limit):mode===2?legacy.encodeBpeWord(text,limit):legacy.encode(text,limit);
        const got=check(await service.encodeText(new p.EncodeTextRequest({tokenizerId:id,text,maxTokens:limit,mode})));
        assert.deepEqual(got.tokens,expected,`${format}/${bpe}/${mode}/${limit}: ${JSON.stringify(text)}`);
        const decoded=check(await service.decodeTokens(new p.DecodeTokensRequest({tokenizerId:id,tokens:[...got.tokens,4294967295]})));
        assert.equal(decoded.text,legacy.decode(expected),`decode ${format}/${bpe}/${mode}`);
        comparisons++;
      }
      assert.deepEqual(check(await service.encodeText(new p.EncodeTextRequest({tokenizerId:id,text:'abc'.repeat(200)}))).tokens,legacy.encode('abc'.repeat(200)));
      const ids=format==='json'?Object.values(json):raw.map((_,i)=>i);
      for(const token of ids) {
        assert.equal(check(await service.decodeTokens(new p.DecodeTokensRequest({tokenizerId:id,tokens:[token]}))).text,legacy.decodeToken(token));
        comparisons++;
      }
      check(await service.releaseTokenizer(new p.TokenizerRef({tokenizerId:id})));
      check(await service.releaseTokenizer(new p.TokenizerRef({tokenizerId:id})));
      assert.notEqual((await service.encodeText(new p.EncodeTextRequest({tokenizerId:id,text:'abc'}))).report.status,0);refusals++;
    }
    for (const options of [{},{vocabularyJson:encoder.encode('{"x":-1}')},{vocabularyJson:encoder.encode('{"x":1,"y":1}')},{vocabularyJson:encoder.encode('{"x":4294967296}')},{vocabularyJson:encoder.encode('{"x":1,}')},{vocabularyBinary:Uint8Array.of(255,255,255,255)},{vocabularyBinary:Uint8Array.of(1,0,0,0,4,0,0,0,1)}]) {
      const got=await service.createTokenizer(new p.CreateTokenizerRequest(options));
      assert.notEqual(got.report.status,0); assert.equal(got.tokenizerId,0n);refusals++;
    }
    const empty=check(await service.createTokenizer(new p.CreateTokenizerRequest({tokens:new p.TokenizerVocabulary()})));
    assert.deepEqual(check(await service.encodeText(new p.EncodeTextRequest({tokenizerId:empty.tokenizerId,text:'abc'}))).tokens,[]);
    assert.equal(check(await service.decodeTokens(new p.DecodeTokensRequest({tokenizerId:empty.tokenizerId,tokens:[0,10]}))).text,'');
    assert.notEqual((await service.encodeText(new p.EncodeTextRequest({tokenizerId:empty.tokenizerId,mode:99}))).report.status,0);refusals++;
    check(await service.releaseTokenizer(new p.TokenizerRef(empty)));
    return {status:'pass',comparisons,refusals,reference:'99dfd8f4bcc66b30da6872852b978a5ff10d22d7:ts/core/Tokenizer.ts'};
  } finally {await host.close();await host2.close();}
}

if(process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
  const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href);
  const {Tokenizer}=await import(pathToFileURL(path.resolve(args.get('--reference'))).href);
  const result=await qualifyTokenizer(api,Tokenizer,pathToFileURL(path.resolve(args.get('--wasm'))));
  console.log(JSON.stringify(result));
  if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify(result,null,2)+'\n');
}
