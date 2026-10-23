#!/usr/bin/env python3
"""Join the retained receipt PTQ comparison matrix and verify its controls."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    ordered=sorted(values)
    return {'n':len(values),'median_ms':statistics.median(values),'p95_ms':ordered[math.ceil(.95*len(values))-1],
        'mean_ms':statistics.mean(values),'min_ms':ordered[0],'max_ms':ordered[-1]}


def operation_rows(rows):
    grouped=defaultdict(lambda:{'count':0,'total_ms':0})
    for row in rows:
        if row.get('domain')!='TRACE_TIME_DOMAIN_HOST':continue
        item=grouped[row['name']]
        item['count']+=int(row.get('count',0));item['total_ms']+=int(row.get('totalNs',0))/1e6
    total=sum(row['total_ms'] for row in grouped.values())
    return [{'operator':name,**row,'share':row['total_ms']/total if total else 0}
        for name,row in sorted(grouped.items(),key=lambda x:-x[1]['total_ms'])]


def item_summary(kind,item,backend):
    calls=[call for request in item['warm'] for call in request['calls']]
    out={'correct':item['correct'],'setup_ms':item['setup_ms'],'cold_first_ms':item['cold']['total_ms'],
        'warm_total':stats([request['total_ms'] for request in item['warm']]),
        'components':item['components'],'public_input_bytes':sum(c['input_bytes'] for c in item['cold']['calls']),
        'public_output_bytes':sum(c['output_bytes'] for c in item['cold']['calls']),
        'signature_sha256':hashlib.sha256(json.dumps(item['cold']['signature']).encode()).hexdigest()}
    if kind=='vqa':
        tokens=[call for call in calls if call['role']=='decoder']
        out['generated_tokens']=len(item['cold']['signature'])-1
        out['decoder_first_token']=stats([c['ms'] for c in tokens if c['position']==0])
        out['decoder_later_tokens']=stats([c['ms'] for c in tokens if c['position']>0])
        out['decoder_by_position']=[{'position':i,**stats([c['ms'] for c in tokens if c['position']==i])}
            for i in sorted({c['position'] for c in tokens})]
    if backend=='native':
        out['operators']={role:operation_rows(p['operators']['rows']) for role,p in item['profiles'].items()}
        out['allocators']={role:p['info'].get('allocators',[]) for role,p in item['profiles'].items()}
        out['trace_dropped_events']={role:int(p['info'].get('events',{}).get('dropped',0)) for role,p in item['profiles'].items()}
        api=defaultdict(list)
        for call in item['api_probe']['calls']:api[f"{call['role']}.{call['method']}"].append(call['ms'])
        out['api_probe']={method:{**stats(values),'total_ms':sum(values)} for method,values in api.items()}
        out['run_wrapper_residual_ms']=item['api_probe']['run_wrapper_residual_ms']
    else:
        # OPERATOR summaries intentionally merge contexts. NODE preserves them;
        # never attribute a merged operator row using its first event's lineage.
        nodes=item['profiles']['nodes']
        if not nodes.get('truncated',False):
            out['operators']={role:operation_rows([r for r in nodes['rows']
                if r['lineage']['contextId']==context]) for role,context in item['context_ids'].items()}
            out['operator_attribution']='complete context-preserving NODE summary'
        elif len(item['context_ids'])==1:
            out['operators']={next(iter(item['context_ids'])):operation_rows(item['profiles']['operators']['rows'])}
            out['operator_attribution']='single context OPERATOR summary'
        else:
            out['operators']={'runtime_aggregate':operation_rows(item['profiles']['operators']['rows'])}
            out['operator_attribution']='OPERATOR is runtime aggregate; truncated NODE rows cannot establish complete per-context totals'
        out['allocators']=item['profiles']['info'].get('allocators',[])
        out['trace_dropped_events']=int(item['profiles']['info'].get('events',{}).get('dropped',0))
        out['profile_execute_wall_ms']=sum(call['ms'] for call in item['profile_request']['calls'])
        out['profile_c_execute_ms']=sum(int(r['totalNs'])/1e6 for r in item['profiles']['operations']['rows'] if r['name']=='Execute')
    return out


def shared_output_hashes(a,b):
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if (x['role'],x['position'])!=(y['role'],y['position']):return False
        names=x['outputs'].keys()&y['outputs'].keys()
        if not names or any(x['outputs'][name]!=y['outputs'][name] for name in names):return False
    return True


def tensor_hashes(package):
    data=(package/'model.safetensors').read_bytes()
    size=int.from_bytes(data[:8],'little');header=json.loads(data[8:8+size]);base=8+size
    return {name:{'dtype':info['dtype'],'shape':info['shape'],
        'sha256':hashlib.sha256(data[base+info['data_offsets'][0]:base+info['data_offsets'][1]]).hexdigest()}
        for name,info in header.items() if name!='__metadata__'}


def validate_storage(base,control):
    graph_a,graph_b=read(base/'graph.json'),read(control/'graph.json')
    other_a=[n for n in graph_a['nodes'] if n['opType']!='PackedLinear']
    other_b=[n for n in graph_b['nodes'] if n['opType']!='PackedLinear']
    assert other_a==other_b,'control changed a non-packed node'
    replaced={node['inputs'][port] for node in graph_a['nodes'] if node['opType']=='PackedLinear' for port in ('weight','scale')}
    a,b=tensor_hashes(base),tensor_hashes(control)
    assert a.keys()==b.keys(),'control changed tensor inventory'
    assert all(a[name]==b[name] for name in a.keys()-replaced),'control changed non-packed tensor bytes'
    return {'non_packed_nodes_identical':True,'non_packed_tensor_bytes_identical':True,
        'changed_banks':len([name for name in replaced if a[name]!=b[name]])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--task-jobs-paused',action='store_true')
    args=p.parse_args();root=args.directory
    summary={'format':'receipt-ptq-profile-comparison/v1','host_quiet_verified':False,
        'task_owned_jobs_paused_during_measurements':args.task_jobs_paused,
        'backend_threads':1,
        'scope':'Two public example inputs, first VQA question per input; controls use case 0. This is not a heldout accuracy measurement.',
        'timing':'Fresh host/context setup and first request separate; two warmups then five samples. Direct native API or one isolated Chrome WASM page, no extension UI/tab/shared-worker queue.',
        'limitations':['Desktop/background work is not excluded; whole-host quiet is unverified.',
            'Native timings include NumPy/owned-output copies; Chrome uses inline protobuf transport.',
            'p95 is nearest rank and equals max for five samples; short sample sets do not establish production-tail latency.',
            'Node/allocator/API probes are separate from warm timing. Dynamic activation preparation remains inside PackedLinear spans.',
            'Allocator coverage is partial; warm reuse is not a claim of allocation-free execution.',
            'Practical I8/group32 control is qualified only on these public benchmark inputs, not the full heldout set.'],
        'runs':{},'primary':[],'controls':{},'artifact_sha256':{},'runtime_hashes':{}}
    raw={};specs={}
    for directory in sorted(root.glob('*-case*')):
        if not directory.is_dir():continue
        label,index=directory.name.rsplit('-case',1);case=int(index);specs[(label,case)]=read(directory/'spec.json')
        for backend in ('native','chrome'):
            path=directory/f'{label}-{backend}.json'
            report=read(path);raw[(label,case,backend)]=report
            if backend=='native':
                if 'cpu_affinity' in summary:assert summary['cpu_affinity']==report['affinity']
                summary['cpu_affinity']=report['affinity']
            summary['artifact_sha256'][str(path.relative_to(root))]=sha(path)
            summary['runtime_hashes'][f'{label}/{backend}']=(report['library_sha256'] if backend=='native' else report['runtime_sha256'])
            if backend=='chrome':summary['chrome_version']=report['browser']
            rows=summary['runs'][f'{label}/case{case}/{backend}']={}
            for model,variants in report['models'].items():
                rows[model]={variant:item_summary(model,item,backend) for variant,item in variants.items()}
                for variant,item in rows[model].items():
                    inventories=specs[(label,case)]['models'][model][variant]['roles']
                    item['storage']={role:c['inventory'] for role,c in inventories.items()}
                    item['package_bytes']=sum(c['inventory']['package_bytes'] for c in inventories.values())
    for case in (0,1):
        for label in ('runtime','mixed'):
            for model in ('digit','vqa'):
                a=specs[('before',case)]['inputs'][model];b=specs[(label,case)]['inputs'][model]
                assert [(v['name'],v['shape'],v['dtype'],v['sha256']) for v in a]==[(v['name'],v['shape'],v['dtype'],v['sha256']) for v in b]
        for backend in ('native','chrome'):
            before=summary['runs'][f'before/case{case}/{backend}'];after=summary['runs'][f'mixed/case{case}/{backend}']
            for model in ('digit','vqa'):
                for variant in ('w8a8','w4a8'):
                    a,b=before[model][variant],after[model][variant]
                    summary['primary'].append({'case':case,'backend':backend,'model':model,'precision':variant,
                        'before_ms':a['warm_total']['median_ms'],'after_ms':b['warm_total']['median_ms'],
                        'speedup':a['warm_total']['median_ms']/b['warm_total']['median_ms'],
                        'before_bytes':a['package_bytes'],'after_bytes':b['package_bytes']})
    for backend in ('native','chrome'):
        controls=summary['controls'][backend]={'runtime_only':[],'same_conv':{},'feedback':{}}
        for case in (0,1):
            for model in ('digit','vqa'):
                a=raw[('before',case,backend)]['models'][model]['w4a8'];b=raw[('runtime',case,backend)]['models'][model]['w4a8']
                original=specs[('before',case)]['models'][model]['w4a8']['roles']
                repeated=specs[('runtime',case)]['models'][model]['w4a8']['roles']
                assert {k:v['inventory']['sha256'] for k,v in original.items()}=={k:v['inventory']['sha256'] for k,v in repeated.items()}
                equal=a['output_sha256']==b['output_sha256'];assert equal,'runtime-only changed numerical outputs'
                controls['runtime_only'].append({'case':case,'model':model,'output_bytes_equal':equal,
                    'before_ms':a['total']['median_ms'],'after_ms':b['total']['median_ms']})
        for model in ('digit','vqa'):
            base=raw[('mixed',0,backend)]['models'][model]['w4a8']
            entry=controls['same_conv'][model]={}
            for variant in ('w4-codes-i8','w8-group32-mse'):
                candidate=raw[('controls',0,backend)]['models'][model][variant]
                record=entry[variant]={'baseline_i4_ms':base['total']['median_ms'],'control_ms':candidate['total']['median_ms'],
                    'output_bytes_equal':base['output_sha256']==candidate['output_sha256'],'public_example_correct':candidate['correct']}
                record['storage_checks']={role:validate_storage(Path(contract['path']),Path(specs[('controls',0)]['models'][model][variant]['roles'][role]['path']))
                    for role,contract in specs[('mixed',0)]['models'][model]['w4a8']['roles'].items()}
                if variant=='w4-codes-i8':assert record['output_bytes_equal'],'lossless I8 changed outputs'
        a=raw[('explicit',0,backend)]['models']['vqa']['w4a8'];b=raw[('mixed',0,backend)]['models']['vqa']['w4a8']
        assert a['cold']['signature']==b['cold']['signature'] and shared_output_hashes(a['output_sha256'],b['output_sha256'])
        controls['feedback']={'signature_equal':True,'common_output_bytes_equal':True,
            'explicit_ms':a['total']['median_ms'],'feedback_ms':b['total']['median_ms'],
            'explicit_public_io_bytes':sum(c['input_bytes']+c['output_bytes'] for c in a['cold']['calls']),
            'feedback_public_io_bytes':sum(c['input_bytes']+c['output_bytes'] for c in b['cold']['calls'])}
        runtime=raw[('runtime',0,backend)]['models']['vqa']['w4a8']
        controls['conv_recipe_only']={'case':0,'old_conv_fp32_ms':runtime['total']['median_ms'],
            'mixed_conv_i8_ms':a['total']['median_ms'],'boundary':'same new runtime; explicit KV transport on both'}
    summary['packed_microbench']={name:read(root/f'packed-breakdown-{name}.json') for name in ('scalar','sse2')}
    summary['timing_caller_sha256']={name:sha(root/'timing-callers'/name) for name in
        ('extension_ptq_profile.py','extension_ptq_profile.mjs','packed_i8_breakdown.c')
        if (root/'timing-callers'/name).is_file()}
    summary['timing_caller_snapshot_present']=len(summary['timing_caller_sha256'])==3
    summary['caller_sha256']={name:sha(Path(__file__).with_name(name)) for name in
        ('extension_ptq_profile.py','extension_ptq_profile.mjs','packed_i8_breakdown.c','summarize_extension_ptq_profile.py')}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for row in summary['primary']:print(row)


if __name__=='__main__':main()
