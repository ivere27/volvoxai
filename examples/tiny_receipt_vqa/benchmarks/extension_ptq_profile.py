#!/usr/bin/env python3
"""Profile installed receipt extensions without UI/worker queue latency.

Run with the extension's Python environment. --models-root may name a retained
archive containing digit/ and vqa/ package roots. Native runs use exactly one CPU
thread; VOLVOXAI_LIBRARY chooses an explicit retained/candidate native binary.
The I8 control losslessly repacks I4 codes, retaining the same Conv precision,
group scales and numerical weights. It is not a practical W8 quantization.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from safetensors.numpy import load_file, save_file
import volvoxai as vx
from google.protobuf.json_format import MessageToDict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(samples):
    return {'n': len(samples), 'mean_ms': statistics.mean(samples), 'median_ms': statistics.median(samples),
            'p95_ms':sorted(samples)[math.ceil(len(samples)*.95)-1],
            'min_ms': min(samples), 'max_ms': max(samples), 'samples_ms': samples}


def load_weight_shards(paths):
    tensors = {}
    for path in paths:
        shard = load_file(path)
        if tensors.keys() & shard.keys():
            raise ValueError(f'duplicate tensor names across weight shards: {path}')
        tensors.update(shard)
    return tensors


def inventory(package, weight_paths=None):
    weight_paths = weight_paths or [package/'model.safetensors']
    graph = json.loads((package/'graph.json').read_text())
    tensors = load_weight_shards(weight_paths)
    owners = defaultdict(set)
    def resolve(name):
        return name if name in tensors else graph.get('banks', {}).get(name, name)
    for node in graph['nodes']:
        for name in node.get('inputs', {}).values():
            resolved = resolve(name)
            if resolved in tensors:
                owners[resolved].add(node['opType'])
    for reference in graph.get('quantization', {}).get('tensors', {}).values():
        for key in ('scale_tensor', 'zero_point_tensor'):
            if key in reference and resolve(reference[key]) in tensors:
                owners[resolve(reference[key])].add('QuantizationMetadata')
    size_by_op = Counter()
    for name, values in tensors.items():
        size_by_op['+'.join(sorted(owners[name])) or 'unused'] += values.nbytes
    files = [package/'graph.json', *weight_paths]
    return {'package_bytes': sum(path.stat().st_size for path in files),
            'byte_scope': 'This role graph plus every loaded shard; shared shards count once per role here. See model package_bytes_unique for storage size.',
            'tensor_bytes_by_consumer_op': dict(size_by_op),
            'nodes_by_op': dict(Counter(node['opType'] for node in graph['nodes'])),
            'sha256': {os.path.relpath(path, package):digest(path) for path in files}}


def repack_i8(source, destination, weight_paths=None):
    graph = json.loads((source/'graph.json').read_text())
    tensors = load_weight_shards(weight_paths or [source/'model.safetensors'])
    shapes={name:value['shape'] for name,value in graph['inputs'].items()}
    for node in graph['nodes']:
        shapes.update({value['tensor']:value['shape'] for value in node['outputs'].values()})
    converted = set()
    for node in graph['nodes']:
        if node['opType'] != 'PackedLinear':
            continue
        assert node['params']['weight_bits'] == 4
        name = node['inputs']['weight']
        if name not in converted:
            original = tensors[name].view(np.uint32)
            n, words = original.shape
            width=shapes[node['inputs']['input']][-1]
            assert isinstance(width,int) and width>0 and words==(width+7)//8
            codes = ((original[:,:,None] >> (np.arange(8,dtype=np.uint32)*4)) & 15).reshape(n,words*8)[:,:width]
            # Preserve every logical code; each format owns its own zero padding.
            encoded=np.full((n,((width+3)//4)*4),128,dtype=np.uint32)
            encoded[:,:width]=codes+120
            tensors[name] = np.bitwise_or.reduce(encoded.reshape(n,-1,4) <<
                (np.arange(4,dtype=np.uint32)*8),axis=2).view(np.int32)
            converted.add(name)
        node['params']['weight_bits'] = 8
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'graph.json').write_text(json.dumps(graph,separators=(',',':')))
    save_file(tensors,destination/'model.safetensors')


def mse_i8_control(source, original, destination, weight_paths=None):
    """Fit I8 from original F32, retaining every non-packed tensor byte exactly."""
    import tempfile
    graph=json.loads((source/'graph.json').read_text())
    original_graph=json.loads((original/'graph.json').read_text())
    tensors=load_weight_shards(weight_paths or [source/'model.safetensors'])
    rules=[vx.pb.QuantizationRule(operator_name=name,keep_float=vx.pb.Empty())
           for name in sorted({n['opType'] for n in original_graph['nodes']}-{'Linear','MatMul'})]
    with tempfile.TemporaryDirectory(prefix='receipt-i8-control-') as temporary:
        fitted=Path(temporary)/'fitted'
        vx.quantize(original,output=fitted,weight_bits=8,group_size=32,algorithm='mse-clip',rules=rules,cpu_threads=1)
        nodes={n['id']:n for n in json.loads((fitted/'graph.json').read_text())['nodes']}
        replacements=load_file(fitted/'model.safetensors')
        for node in graph['nodes']:
            if node['opType']!='PackedLinear':continue
            matched=nodes[node['id']]
            assert matched['opType']=='PackedLinear'
            assert matched['params']['group_size']==node['params']['group_size']==32
            for port in ('weight','scale'):
                tensors[node['inputs'][port]]=replacements[matched['inputs'][port]]
            node['params']['weight_bits']=8
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'graph.json').write_text(json.dumps(graph,separators=(',',':')))
    save_file(tensors,destination/'model.safetensors')


def unpack_conv_i8_control(source, destination, weight_paths=None):
    """Preserve every Conv4 code/scale in ordinary I8 OHWI storage."""
    graph = json.loads((source/'graph.json').read_text())
    tensors = load_weight_shards(weight_paths or [source/'model.safetensors'])
    shapes = {name: value['shape'] for name, value in graph['inputs'].items()}
    for node in graph['nodes']:
        shapes.update({value['tensor']: value['shape'] for value in node['outputs'].values()})
    converted = set()
    for node in graph['nodes']:
        if node['opType'] != 'QConv2D' or node['params'].get('weight_bits', 8) != 4:
            continue
        name = node['inputs']['weight']
        kh, kw = node['params']['kernel_size']
        channels = shapes[node['inputs']['input']][-1]
        groups = node['params'].get('groups', 1)
        assert isinstance(channels, int) and channels > 0 and channels % groups == 0
        width = kh * kw * (channels // groups)
        if name not in converted:
            packed = tensors[name]
            assert packed.dtype == np.uint8 and packed.ndim == 2
            assert packed.shape[1] == (width + 1) // 2
            if width % 2:
                assert np.all(packed[:, -1] >> 4 == 8)
            codes = ((packed[:, :, None] >> np.asarray([0, 4], np.uint8)) & 15)
            centered = codes.reshape(len(packed), -1)[:, :width].astype(np.int8) - 8
            tensors[name] = np.ascontiguousarray(centered.reshape(len(packed), kh, kw, channels // groups))
            zero_name = graph['quantization']['tensors'][name]['zero_point_tensor']
            assert np.all(tensors[zero_name] == 8)
            tensors[zero_name] = np.zeros_like(tensors[zero_name], dtype=np.int8)
            converted.add(name)
        del node['params']['weight_bits'], node['params']['kernel_size']
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'graph.json').write_text(json.dumps(graph, separators=(',', ':')))
    save_file(tensors, destination/'model.safetensors')


def trace_report(trace):
    return {'info':MessageToDict(trace.info),
            'operators':MessageToDict(trace.summary('operator')),
            'activities':MessageToDict(trace.summary('activity')),
            'operations':MessageToDict(trace.summary('operation')),
            'nodes':MessageToDict(trace.summary('node', max_rows=4096))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extensions',type=Path,required=True)
    p.add_argument('--models-root',type=Path)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--label',default='current')
    p.add_argument('--case-index',type=int,default=0,choices=(0,1))
    p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--models',nargs='+',default=['digit','vqa'])
    p.add_argument('--variants',nargs='+',default=['w8a8','w4a8','w4-codes-i8'])
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--feedback',action='store_true')
    p.add_argument('--packed-arithmetic', choices=('dynamic-i8', 'f32', 'f16'),
                   help='Override PackedLinear arithmetic for a developer comparison; weight storage is unchanged.')
    args=p.parse_args()
    if args.repeat<1 or args.warmup<0:p.error('repeat must be positive and warmup nonnegative')
    args.work.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.extensions))
    from shared.model_support import preprocess as digit_preprocess
    from vqa.model import preprocess, load_tokenizer, encoder_feed, decoder_feed, advance, CROSS, PAST, PRESENT, role_weight_paths
    pb=vx.pb
    spec={'label':args.label,'models':{},'inputs':{},'expected':{},'repeat':args.repeat,'warmup':args.warmup,
          'feedback':args.feedback,'case_index':args.case_index}
    for kind in args.models:
        config=json.loads((args.extensions/kind/'extension.json').read_text())
        source=args.extensions/'.cache'/kind/'huggingface'/config['model']['revision']
        root=args.models_root/kind if args.models_root else args.extensions/'.cache'/kind/'packages'/'0.8.0'
        if kind=='digit':
            example=json.loads((source/'examples/examples.json').read_text())[args.case_index]
            image=digit_preprocess(source/'examples'/example['image'])
            feed={'input0':image}
            spec['expected'][kind]=example['expected_record']
        else:
            example=json.loads((source/'examples/family_examples.json').read_text())['examples'][args.case_index]
            question=example['cases'][0]['question']
            tokenizer=load_tokenizer(source)
            feed=encoder_feed(preprocess(source/'examples'/example['image']),tokenizer.encode(question,add_eos=True,max_len=192))
            spec['question']=question
            spec['vocab']=str(root/'ptq/w4a8/vocab.json')
            spec['tokenizer']=str(args.extensions/'vqa/tokenizer.js')
            spec['expected'][kind]=str(example['cases'][0]['expected_answer'])
        spec['inputs'][kind]=[]
        for name,value in feed.items():
            filename=f'{kind}-{name}.bin'
            (args.work/filename).write_bytes(value.tobytes())
            spec['inputs'][kind].append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),
                'file':str(args.work/filename),'sha256':digest(args.work/filename)})
        spec['models'][kind]={}
        for variant in args.variants:
            control=variant in ('w4-codes-i8','w8-group32-mse','conv4-codes-i8')
            base_variant = ('w4a8-conv4' if variant == 'conv4-codes-i8' else
                            'w4a8' if control else variant)
            package=root/'ptq'/base_variant
            roles=['digit'] if kind=='digit' else ['encoder','decoder']
            manifest=json.loads((package/'manifest.json').read_text())
            row={'arithmetic':args.packed_arithmetic or ('f32' if variant=='w8a8' else 'dynamic-i8'),'roles':{}}
            model_files = set()
            for role in roles:
                role_path=package if kind=='digit' else package/role
                weight_paths = [role_path/'model.safetensors'] if kind=='digit' else role_weight_paths(package, role, manifest)
                if variant=='w4-codes-i8':
                    target=args.work/'controls'/kind/role
                    repack_i8(role_path,target,weight_paths)
                    role_path=target
                    weight_paths=[target/'model.safetensors']
                elif variant=='w8-group32-mse':
                    target=args.work/'mse-controls'/kind/role
                    original=root/'fp32-prepared'
                    mse_i8_control(role_path,original if kind=='digit' else original/role,target,weight_paths)
                    role_path=target
                    weight_paths=[target/'model.safetensors']
                elif variant=='conv4-codes-i8':
                    target=args.work/'conv-controls'/kind/role
                    unpack_conv_i8_control(role_path,target,weight_paths)
                    role_path=target
                    weight_paths=[target/'model.safetensors']
                contract=manifest.get('graphs',{}).get(role,{})
                row['roles'][role]={'path':str(role_path),'inputs':contract.get('inputs',{'input0':'input0'}),
                    'weight_paths':[str(path) for path in weight_paths],
                    'inventory':inventory(role_path,weight_paths)}
                model_files.update([role_path/'graph.json', *weight_paths])
            row['package_bytes_unique']=sum(path.stat().st_size for path in model_files)
            row['package_file_sha256']={str(path):digest(path) for path in sorted(model_files)}
            spec['models'][kind][variant]=row
    (args.work/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    if args.prepare_only:
        return
    from volvoxai._library import find_library
    report={'label':args.label,'backend':'cpu','cpu_threads':1,'affinity':sorted(os.sched_getaffinity(0)),
        'cpu_isa_override':os.environ.get('VOLVOXAI_CPU_ISA'),
        'library_sha256':digest(find_library()),'spec_sha256':digest(args.work/'spec.json'),
        'caller_sha256':digest(Path(__file__)),'models':{},
        'timing_boundary':'unprofiled full Python InferenceSession.run including input/output transport; separate node traces',
        'cold_definition':'new module/model/context setup plus first request; OS file cache not flushed',
        'feedback':args.feedback,'case_index':args.case_index}
    for kind,variants in spec['models'].items():
        feed={item['name']:np.fromfile(item['file'],dtype=item['dtype']).reshape(item['shape']) for item in spec['inputs'][kind]}
        report['models'][kind]={}
        for variant,model in variants.items():
            result={'inventory':{role:c['inventory'] for role,c in model['roles'].items()},
                    'packed_arithmetic':model['arithmetic'],'package_bytes_unique':model['package_bytes_unique']}
            report['models'][kind][variant]=result
            started=time.perf_counter_ns()
            sessions={role:vx.InferenceSession(c['path'],backend='cpu',cpu_threads=1,
                weights=c['weight_paths'],packed_arithmetic=model['arithmetic']) for role,c in model['roles'].items()}
            expected_arithmetic=pb.PackedArithmetic.Value('PACKED_ARITHMETIC_'+model['arithmetic'].upper().replace('-','_'))
            assert all(s.backend=='cpu' and s.packed_arithmetic==expected_arithmetic for s in sessions.values())
            result['setup_ms']=(time.perf_counter_ns()-started)/1e6
            observed=[]
            verification=None
            def run(role,values,position=0):
                contract=model['roles'][role]['inputs']
                names=(['slot_logits'] if role=='digit' else
                       ['memory_padding_mask','selected_family_ids',*CROSS] if role=='encoder' else
                       ['logits','present_padding_mask',*PRESENT])
                reuse=[]
                feedback={}
                if args.feedback and role=='decoder':
                    names=['logits']
                    if position:
                        reuse=[contract[name] for name in [*CROSS,'memory_padding_mask','family_ids']]
                        feedback={contract[past]:present for past,present in zip(PAST,PRESENT)}
                        feedback[contract['past_padding_mask']]='present_padding_mask'
                        values={name:values[name] for name in ['decoder_input_ids','position_ids']}
                mapped={contract.get(name,name):np.ascontiguousarray(value) for name,value in values.items()}
                t=time.perf_counter_ns()
                outputs=sessions[role].run(mapped,output_names=names,reuse_inputs=reuse,feedback=feedback)
                elapsed=(time.perf_counter_ns()-t)/1e6
                observed.append({'role':role,'position':position,'ms':elapsed,
                    'input_bytes':sum(v.nbytes for v in values.values()),'output_bytes':sum(v.nbytes for v in outputs.values())})
                if verification is not None:
                    verification.append({'role':role,'position':position,'outputs':{
                        name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in outputs.items()}})
                return outputs
            def request():
                observed.clear()
                t=time.perf_counter_ns()
                if kind=='digit':
                    output=run('digit',feed)
                    signature=output['slot_logits'].argmax(axis=-1).tolist()
                else:
                    encoded=run('encoder',feed)
                    values=decoder_feed(encoded)
                    signature=[1]
                    for pos in range(191):
                        output=run('decoder',values,pos)
                        token=int(output['logits'][0,0].argmax())
                        signature.append(token)
                        if token==2: break
                        if args.feedback:
                            values={**values,'decoder_input_ids':np.asarray([[token]],np.int32),
                                'position_ids':np.asarray([pos+1],np.int32)}
                        else: values=advance(values,output,token,pos)
                    assert signature[-1]==2
                return {'total_ms':(time.perf_counter_ns()-t)/1e6,'calls':list(observed),'signature':signature}
            try:
                result['cold']=request()
                for _ in range(args.warmup):request()
                result['warm']=[request() for _ in range(args.repeat)]
                assert all(r['signature']==result['cold']['signature'] for r in result['warm'])
                result['total']=stats([r['total_ms'] for r in result['warm']])
                result['components']={role:stats([sum(c['ms'] for c in r['calls'] if c['role']==role) for r in result['warm']]) for role in sessions}
                verification=[]
                request()
                result['output_sha256']=verification
                verification=None
                # This separate probe times generated API calls inside run();
                # CopyTensors includes dispatch and copies, not memcpy alone.
                api_calls=[]
                restore=[]
                try:
                    for role,session in sessions.items():
                        for client,method in ((session._engine,'execute'),
                                (session._buffers,'copy_tensors'),(session._buffers,'release_buffers')):
                            original=getattr(client,method)
                            restore.append((client,method,original))
                            def measured(*values,_call=original,_role=role,_method=method,**keywords):
                                t=time.perf_counter_ns()
                                try:return _call(*values,**keywords)
                                finally:api_calls.append({'role':_role,'method':_method,'ms':(time.perf_counter_ns()-t)/1e6})
                            setattr(client,method,measured)
                    probe=request()
                finally:
                    for client,method,original in restore:setattr(client,method,original)
                result['api_probe']={'calls':api_calls,'request':probe,
                    'run_wrapper_residual_ms':sum(c['ms'] for c in probe['calls'])-sum(c['ms'] for c in api_calls),
                    'boundary':'separate request; generated API wall time includes dispatch/codec overhead; wrapper residual includes NumPy allocations and binding validation'}
                result['profiles']={}
                for role,session in sessions.items():
                    # Capture a complete real request, not a synthetic final token.
                    with session.trace(detail='nodes',memory=True,execution_plans=True,capacity_bytes=32*1024*1024) as trace:
                        traced_request=request()
                    result['profiles'][role]=trace_report(trace)
                    result['profiles'][role]['request']=traced_request
                    trace.export_chrome_trace(args.work/f'{args.label}-{kind}-{variant}-{role}-trace.json')
                if kind=='vqa':
                    import re
                    result['text']=tokenizer.decode(result['cold']['signature'])
                    matched=re.search(r'<answer>([^<>]*)</answer>',result['text'])
                    result['correct']=bool(matched and matched[1]==spec['expected'][kind])
                else:
                    slots=result['cold']['signature'][0]
                    def digits(items):
                        return ''.join(str(n) for n in items[:items.index(10) if 10 in items else len(items)])
                    result['correct']={'phone':digits(slots[:12]),'street':digits(slots[12:])}==spec['expected'][kind]
                assert result['correct'],f'{kind}/{variant}: public example accuracy failure'
                print(kind,variant,result['total']['median_ms'],flush=True)
            finally:
                for session in sessions.values():session.close()
            (args.work/f'{args.label}-native.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
