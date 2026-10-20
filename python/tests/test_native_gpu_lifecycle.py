"""Native GPU ownership and attention policy on explicitly selected hardware.

Set VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda/vulkan/opengl on an idle device.
Subprocesses catch driver TLS crashes after an owner and its workers retire.
CrossSDPA runs numerical checks on CUDA; Vulkan/OpenGL must retain their
typed public admission refusal, despite sharing the internal MHA shader.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = os.environ.get("VOLVOXAI_TEST_NATIVE_GPU_BACKEND")

CHILD = textwrap.dedent("""\
    import json,struct,sys,time
    from pathlib import Path
    sys.path.insert(0,str(Path.cwd()/'python'))
    import volvoxai as vx
    pb=vx.pb
    backend,=sys.argv[1:]
    graph=json.dumps({'format':'volvox-graph/v1','dimensions':{},
        'inputs':{'input':{'shape':[1,2],'dtype':'float32'}},
        'nodes':[{'id':'linear','opType':'Linear',
            'inputs':{'input':'input','weight':'weight','bias':'bias'},
            'outputs':{'out':{'tensor':'output','shape':[1,2],'dtype':'float32'}},
            'params':{'weight_layout':'dout_din'}}],
        'outputs':['output'], 'execution':{'stateless':{},'entrypoints':[{'name':'evaluate','effect':1,'allowed_ranges':[1],'inputs':['input'],'outputs':['output']}]}}).encode()
    header=json.dumps({'weight':{'dtype':'F32','shape':[2,2],'data_offsets':[0,16]},
                       'bias':{'dtype':'F32','shape':[2],'data_offsets':[16,24]}},
                      separators=(',',':')).encode()
    header+=b' '*((-len(header))%8)
    weights=struct.pack('<Q',len(header))+header+struct.pack('<6f',1,2,-3,4,.25,-.5)
    for iteration in range(4):
        host=vx.open_library()
        try:
            client=vx.VxInferenceServiceClient(host)
            runtime=client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            model=client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=pb.ModelPackage(graph_document=graph,weight_shards=[weights])))
            compiled=client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=[backend],operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            assert compiled.report.route.attested and compiled.report.backend==backend
            context=client.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id,graph=pb.GraphExecution()))
            for values,expected in [((-2,3),(4.25,17.5)),((.5,-1),(-1.25,-6))]:
                result=client.execute(pb.ExecuteRequest(context_id=context.context_id,
                    scope=pb.ExecutionScope(entrypoint='evaluate',full=pb.Empty()),
                    inputs=pb.InputBindings(values=[pb.Tensor(name='input',shape=[1,2],dtype=pb.DataType.DATA_TYPE_F32,
                                      inline=struct.pack('<2f',*values))])))
                assert result.report.route.attested and result.report.backend==backend
                output=next(tensor for tensor in result.outputs if tensor.name=='output')
                assert list(output.shape)==[1,2] and output.dtype==pb.DataType.DATA_TYPE_F32
                actual=struct.unpack('<2f',output.inline)
                assert all(abs(a-b)<1e-5 for a,b in zip(actual,expected)),(actual,expected)
        finally:
            host.close()
        time.sleep(.01)
    print(backend,'four owners and eight independent numerical checks passed')
""")

ATTENTION_CHILD = textwrap.dedent("""\
    import json,math,struct,sys
    from pathlib import Path
    sys.path.insert(0,str(Path.cwd()/'python'))
    import volvoxai as vx
    pb=vx.pb
    backend,=sys.argv[1:]
    batch,queries,keys,heads=2,2,3,3
    def f32(values):
        return struct.unpack('<'+'f'*len(values),struct.pack('<'+'f'*len(values),*values))
    def tensor(name,shape,values,integer=False):
        return pb.Tensor(name=name,shape=shape,
            dtype=pb.DataType.DATA_TYPE_I32 if integer else pb.DataType.DATA_TYPE_F32,
            inline=struct.pack('<'+('i' if integer else 'f')*len(values),*values))
    def route(report):
        assert report.route.attested and report.backend==backend,report
        assert report.route.fallback_nodes==0 and report.route.missing_nodes==0,report
        assert not (report.fallback and report.fallback.operator_fallback_used),report
    def reference(q,k,v,width,keep):
        result=[0.0]*len(q)
        for b in range(batch):
            for row in range(queries):
                for head in range(heads):
                    visible=[key for key in range(keys) if keep is None or keep[key]]
                    if not visible:continue
                    scores=[sum(q[((b*queries+row)*heads+head)*width+d]*
                                k[((b*keys+key)*heads+head)*width+d] for d in range(width))/math.sqrt(width)
                            for key in visible]
                    largest=max(scores)
                    weights=[math.exp(score-largest) for score in scores]
                    total=sum(weights)
                    for d in range(width):
                        result[((b*queries+row)*heads+head)*width+d]=sum(
                            weight/total*v[((b*keys+key)*heads+head)*width+d]
                            for weight,key in zip(weights,visible))
        return result
    host=vx.open_library()
    try:
        client=vx.VxInferenceServiceClient(host)
        runtime=client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
        for width in (8,64):
            q=f32([.5*math.sin(i*.17+.3) for i in range(batch*queries*heads*width)])
            k=f32([.4*math.cos(i*.13+.7) for i in range(batch*keys*heads*width)])
            v=f32([50.0+head if key==1 else .2+.3*head+.07*key+.01*d+.03*b
                   for b in range(batch) for key in range(keys) for head in range(heads) for d in range(width)])
            for masked in (False,True):
                inputs={name:{'shape':shape,'dtype':'float32'} for name,shape in
                    [('q',[batch,queries,heads*width]),('k',[batch,keys,heads*width]),('v',[batch,keys,heads*width])]}
                ports={name:name for name in ('q','k','v')}
                if masked:
                    inputs['keep']={'shape':[keys],'dtype':'int32'};ports['mask']='keep'
                # Omitted kv_heads preserves the legacy MHA contract.
                graph={'format':'volvox-graph/v1','dimensions':{},'inputs':inputs,
                    'nodes':[{'id':'attention','opType':'CrossSDPA','inputs':ports,
                        'outputs':{'out':{'tensor':'y','shape':[batch,queries,heads*width],'dtype':'float32'}},
                        'params':{'heads':heads,'causal':False}}], 'outputs':['y'],
                    'execution':{'stateless':{},'entrypoints':[{'name':'evaluate','effect':1,
                        'allowed_ranges':[1],'inputs':list(inputs),'outputs':['y']}]}}
                model=client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=pb.ModelPackage(graph_document=json.dumps(graph).encode())))
                request=pb.CompileModelRequest(model_id=model.model_id,
                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                        backends=[backend],operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID))
                if backend in ('vulkan','opengl'):
                    try:
                        client.compile_model(request)
                    except vx.VolvoxAIError as error:
                        assert error.status==pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED,error.report
                        assert error.code==pb.OperationCode.OPERATION_CODE_BOUNDED_DOMAIN_UNSUPPORTED,error.report
                        assert error.stage==pb.OperationStage.OPERATION_STAGE_COMPILE,error.report
                        assert error.backend==backend,error.report
                        assert 'missing-native-gpu-domain-proof' in error.report.message,error.report
                        assert 'CrossSDPA' in error.report.message,error.report
                        assert not error.response.compiled_model_id,error.response
                    else:
                        raise AssertionError('CrossSDPA unexpectedly admitted without a native domain proof')
                    client.release_model(pb.ModelRef(model_id=model.model_id))
                    continue
                compiled=client.compile_model(request)
                route(compiled.report)
                context=client.create_execution_context(pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id,graph=pb.GraphExecution()))
                for keep in ([None] if not masked else [[1,0,1],[0,0,0]]):
                    values=[tensor('q',[batch,queries,heads*width],q),
                        tensor('k',[batch,keys,heads*width],k),tensor('v',[batch,keys,heads*width],v)]
                    if keep is not None:values.append(tensor('keep',[keys],keep,True))
                    packet=client.execute(pb.ExecuteRequest(context_id=context.context_id,
                        scope=pb.ExecutionScope(entrypoint='evaluate',full=pb.Empty()),
                        inputs=pb.InputBindings(values=values)))
                    route(packet.report)
                    output=next(value for value in packet.outputs if value.name=='y')
                    assert list(output.shape)==[batch,queries,heads*width] and output.dtype==pb.DataType.DATA_TYPE_F32
                    actual=struct.unpack('<'+'f'*len(q),output.inline)
                    expected=reference(q,k,v,width,keep)
                    for index,(a,b) in enumerate(zip(actual,expected)):
                        assert math.isfinite(a) and abs(a-b)<=3e-5*(1+abs(b)),(width,masked,keep,index,a,b)
                    if keep==[0,0,0]:assert all(value==0.0 for value in actual)
                client.release_execution_context(pb.ExecutionContextRef(context_id=context.context_id))
                client.release_compiled_model(pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
                client.release_model(pb.ModelRef(model_id=model.model_id))
        client.release_runtime(pb.RuntimeRef(runtime_id=runtime.runtime_id))
    finally:
        host.close()
    if backend in ('vulkan','opengl'):
        print(backend,'CrossSDPA D8/D64 none/K: four typed admission refusals passed; no execution claimed')
    else:
        print(backend,'CrossSDPA MHA D8/D64 none/K/all-masked: four graphs, six independent numerical checks passed')
""")


@unittest.skipUnless(BACKEND, "set VOLVOXAI_TEST_NATIVE_GPU_BACKEND on physical hardware")
class NativeGpuLifecycleTest(unittest.TestCase):
    def test_owner_lifecycle(self):
        self.assertIn(BACKEND, ("cuda", "vulkan", "opengl"))
        completed = subprocess.run([sys.executable, "-c", CHILD, BACKEND],
                                   cwd=ROOT, text=True, capture_output=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_cross_sdpa_mha_numerics_or_typed_admission_refusal(self):
        self.assertIn(BACKEND, ("cuda", "vulkan", "opengl"))
        completed = subprocess.run([sys.executable, "-c", ATTENTION_CHILD, BACKEND],
                                   cwd=ROOT, text=True, capture_output=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
