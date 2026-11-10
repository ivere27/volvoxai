#!/usr/bin/env python3
"""Replay TTS validation corpus with a required native backend and no fallback."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--backend',choices=['cpu','cuda','vulkan','opengl','metal'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); results=[]
    receipt={'backend':args.backend,'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),
             'corpus_sha256':hashlib.sha256(args.corpus.read_bytes()).hexdigest(),'results':results,'status':'FAIL'}
    try:
        cases=json.loads(args.corpus.read_text())['cases']
        assert isinstance(cases,list) and cases, 'nonempty validation corpus required'
        for case in cases:
            with tempfile.TemporaryDirectory(prefix='volvox-tts-case-') as directory:
                root=Path(directory); (root/'graph.json').write_text(json.dumps(case['graph']))
                command=[str(args.binary.resolve()),'run',str(root/'graph.json'),'--'+args.backend,'--report-json',str(root/'report.json')]
                for index,shard in enumerate(case['weight_shards']):
                    path=root/f'weights{index}.safetensors'; path.write_bytes(base64.b64decode(shard)); command+=['--weights',str(path)]
                for index,tensor in enumerate(case['inputs']):
                    path=root/f"in{index}.{'i32' if tensor['dtype']=='I32' else 'f32'}"; path.write_bytes(base64.b64decode(tensor['data']))
                    shape=f"[{','.join(map(str,tensor['shape']))}]" if tensor['shape'] else ''
                    command+=['--input',f"{tensor['name']}{shape}={path}"]
                for index,tensor in enumerate(case['expected']): command+=['--output',f"{tensor['name']}={root}/out{index}.{'i32' if tensor['dtype']=='I32' else 'f32'}"]
                run=subprocess.run(command,capture_output=True,text=True,timeout=120)
                if run.returncode: raise RuntimeError(f"{case['name']}: {run.stderr}")
                report=json.loads((root/'report.json').read_text())
                for phase in ('compile','execute'):
                    operation=report[phase]; route=operation['route']
                    assert operation['backend']==args.backend,operation
                    assert route['builtin'] and route['attested'] and route['selectedNodes']==route['activeNodes'] and not route['missingNodes'] and not route['fallbackNodes'],route
                maximum=0.
                for index,tensor in enumerate(case['expected']):
                    expected=base64.b64decode(tensor['data']); actual=(root/f"out{index}.{'i32' if tensor['dtype']=='I32' else 'f32'}").read_bytes()
                    assert len(actual)==len(expected),(case['name'],'byte length')
                    if case.get('bitwise'): assert actual==expected,(case['name'],'exact output bits')
                    code='i' if tensor['dtype']=='I32' else 'f'
                    for (a,),(b,) in zip(struct.iter_unpack('<'+code,actual),struct.iter_unpack('<'+code,expected)):
                        if not math.isfinite(b): assert (math.isnan(a) if math.isnan(b) else a==b),(case['name'],a,b)
                        else:
                            error=abs(a-b); maximum=max(maximum,error)
                            assert math.isfinite(a) and error <= (0 if code=='i' else 1e-5+1e-5*abs(b)),(case['name'],a,b)
                results.append({'name':case['name'],'max_error':maximum,'route':route,'device':operation.get('device'),'pass':True})
                print('PASS',args.backend,case['name'],flush=True)
        receipt['status']='PASS'
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__': main()
