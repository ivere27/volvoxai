"""Numerical native GPU ownership checks on explicitly selected hardware.

Set VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda/vulkan/opengl on an idle device.
Subprocesses catch driver TLS crashes after an owner and its workers retire.
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
        'outputs':['output']}).encode()
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
            for values,expected in [((-2,3),(4.25,17.5)),((.5,-1),(-1.25,-6))]:
                result=client.run(pb.RunRequest(compiled_model_id=compiled.compiled_model_id,
                    inputs=[pb.Tensor(name='input',shape=[1,2],dtype=pb.DataType.DATA_TYPE_F32,
                                      inline=struct.pack('<2f',*values))]))
                try:
                    assert result.report.route.attested and result.report.backend==backend
                    deadline=time.monotonic()+30
                    while True:
                        info=client.get_result(pb.ResultRef(result_id=result.result_id))
                        if info.state!=pb.ResultState.RESULT_STATE_PENDING:
                            assert info.state==pb.ResultState.RESULT_STATE_READY
                            break
                        assert time.monotonic()<deadline
                        time.sleep(.001)
                    output=client.read_output(pb.ReadOutputRequest(
                        result_id=result.result_id,name='output')).tensor
                    assert list(output.shape)==[1,2] and output.dtype==pb.DataType.DATA_TYPE_F32
                    actual=struct.unpack('<2f',output.inline)
                    assert all(abs(a-b)<1e-5 for a,b in zip(actual,expected)),(actual,expected)
                finally:
                    client.release_result(pb.ResultRef(result_id=result.result_id))
        finally:
            host.close()
        time.sleep(.01)
    print(backend,'four owners and eight independent numerical checks passed')
""")


@unittest.skipUnless(BACKEND, "set VOLVOXAI_TEST_NATIVE_GPU_BACKEND on physical hardware")
class NativeGpuLifecycleTest(unittest.TestCase):
    def test_owner_lifecycle(self):
        self.assertIn(BACKEND, ("cuda", "vulkan", "opengl"))
        completed = subprocess.run([sys.executable, "-c", CHILD, BACKEND],
                                   cwd=ROOT, text=True, capture_output=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
