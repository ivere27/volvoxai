#!/usr/bin/env python3
"""Build immutable TTS/vocoder and scan oracles for every execution backend."""
from __future__ import annotations
import argparse
import base64
import json
import math
from pathlib import Path
import tempfile
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tools.export_safetensors import export_onnx_model
from tools.tts_vocoder_fixtures import build_tts_vocoder_fixture, build_kokoro_fixture


def encoded(name, value):
    value = np.ascontiguousarray(value)
    return dict(name=name, shape=list(value.shape), dtype='I32' if value.dtype == np.int32 else 'F32',
                data=base64.b64encode(value.tobytes()).decode())


def fixtures():
    yield 'tts_vocoder', (*build_tts_vocoder_fixture(), {})
    yield 'kokoro_norm_snake', (*build_kokoro_fixture(), {})
    # TTS normalization broadcasts one live scalar per channel along time.
    # Include mixed leading/trailing axes and an odd long fill, with exact bit
    # comparison so signed zero and NaN payloads cannot be normalized away.
    patterns = np.array([0, 0x80000000, 0x7fc01234, 0x7f800000,
                         0xff800000, 0xffffffff, 1, 0x00800000], np.uint32)
    for dtype in (np.float32, np.int32):
        for index, (shape, target) in enumerate([
                ((1, 8, 1), (1, 8, 7921)), ((2, 1, 3, 1), (2, 4, 3, 17)),
                ((1, 2, 1, 1), (3, 2, 5, 7))]):
            x = np.resize(patterns, int(np.prod(shape))).view(dtype).reshape(shape)
            kind = TensorProto.FLOAT if dtype == np.float32 else TensorProto.INT32
            graph = helper.make_graph([helper.make_node('Expand', ['x', 'shape'], ['y'])],
                'trailing_broadcast', [helper.make_tensor_value_info('x', kind, shape)],
                [helper.make_tensor_value_info('y', kind, target)],
                [numpy_helper.from_array(np.asarray(target, np.int64), 'shape')])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
            options = {'output_dtypes': {'y': 'int32'}} if dtype == np.int32 else {}
            yield f'expand_fill_{np.dtype(dtype).name}_{index}', (
                model, {'x': x}, {'y': np.broadcast_to(x, target).copy()}, options)
    for op in ('Exp', 'Sin', 'Cos'):
        x = np.array([-90, -3, -0., 0., .1, 3, 88, 88.5, 89, np.inf, -np.inf, np.nan], np.float32) if op == 'Exp' else np.linspace(-32, 32, 37, dtype=np.float32)
        graph = helper.make_graph([helper.make_node(op, ['x'], ['y'])], op,
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, x.shape)],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, x.shape)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',17)])
        with np.errstate(over='ignore',invalid='ignore'):
            expected = {'y': getattr(np,op.lower())(x)}
        yield op.lower(), (model, {'x': x}, expected, {})
    unary = {"Relu": lambda x: np.maximum(x, 0), "Sqrt": np.sqrt, "Log": np.log, "Erf": np.vectorize(math.erf),
             "Floor": np.floor, "Ceil": np.ceil, "Round": np.rint, "Atan": np.arctan,
             "Softplus": lambda x: np.logaddexp(0, x), "Neg": np.negative}
    for op, reference in unary.items():
        x = np.array([-100, -40, -5, -2.5, -1.5, -.5, -.001, -0., 0., .001,
            .5, 1, 1.5, 2.5, 5, 40, 100, np.inf, -np.inf, np.nan], np.float32)
        if op == "Relu": x = x[~np.isnan(x)]
        graph = helper.make_graph([helper.make_node(op, ["x"], ["y"])], op,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, x.shape)])
        with np.errstate(all="ignore"):
            expected = reference(x.astype(np.float64)).astype(np.float32)
        yield op.lower(), (helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
                            {"x": x}, {"y": expected}, {})
    left = np.array([[-1], [0], [np.nan]], np.float32)
    right = np.array([[-np.inf, -0., 1, np.inf, np.nan]], np.float32)
    comparisons = {"Equal": np.equal, "GreaterOrEqual": np.greater_equal, "Greater": np.greater,
                   "Less": np.less, "LessOrEqual": np.less_equal}
    nodes = [helper.make_node(op, ["a", "b"], [op]) for op in comparisons]
    nodes.append(helper.make_node("And", ["GreaterOrEqual", "LessOrEqual"], ["both"]))
    expected = {op: reference(left, right).astype(np.int32) for op, reference in comparisons.items()}
    expected["both"] = np.logical_and(expected["GreaterOrEqual"], expected["LessOrEqual"]).astype(np.int32)
    graph = helper.make_graph(nodes, "float_comparisons",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, left.shape),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, right.shape)],
        [helper.make_tensor_value_info(name, TensorProto.BOOL, [3,5]) for name in expected])
    yield "float_comparisons", (helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        {"a": left, "b": right}, expected, {"output_dtypes": {name: "int32" for name in expected}})
    # Metadata grows with the graph; this exceeds both obsolete exporter caps.
    count = 2052
    nodes = [helper.make_node("Neg", ["x" if i == 0 else f"y{i-1}"], [f"y{i}"]) for i in range(count)]
    x = np.array([-2.5, -0., 3.25], np.float32)
    graph = helper.make_graph(nodes, "large_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
        [helper.make_tensor_value_info(f"y{count-1}", TensorProto.FLOAT, x.shape)])
    yield "large_graph", (helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        {"x": x}, {"y": x.copy()}, {})
    for dtype in (np.float32, np.int32):
        for axis in range(3):
            for exclusive in (0, 1):
                for reverse in (0, 1):
                    x = (np.arange(30).reshape(2,3,5) % 9 - 4).astype(dtype)
                    if dtype == np.int32:
                        x[:,1,:] = np.iinfo(np.int32).max
                    else:
                        x /= np.float32(7)
                    value = np.flip(x, axis) if reverse else x
                    expected = np.cumsum(value, axis=axis, dtype=dtype)
                    if exclusive:
                        expected = np.roll(expected, 1, axis=axis)
                        at = [slice(None)]*3; at[axis] = 0
                        expected[tuple(at)] = 0
                    if reverse: expected = np.flip(expected, axis)
                    tensor_type = TensorProto.INT32 if dtype == np.int32 else TensorProto.FLOAT
                    graph = helper.make_graph([helper.make_node('CumSum',['x','axis'],['y'],exclusive=exclusive,reverse=reverse)], 'scan',
                        [helper.make_tensor_value_info('x',tensor_type,x.shape)],
                        [helper.make_tensor_value_info('y',tensor_type,x.shape)],
                        [numpy_helper.from_array(np.array(axis-3,np.int64),'axis')])
                    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
                    name = f'cumsum_{np.dtype(dtype).name}_{axis}_{exclusive}_{reverse}'
                    yield name, (model, {'x':x}, {'y':np.ascontiguousarray(expected)}, {})
    # The same bounded descriptor is replayed at a concrete interior extent.
    for op in ('Exp','CumSum'):
        x = np.arange(9,dtype=np.float32).reshape(3,3)/8
        nodes = [helper.make_node(op,['x','axis'] if op=='CumSum' else ['x'],['y'])]
        constants = [numpy_helper.from_array(np.array(0,np.int64),'axis')] if op=='CumSum' else []
        graph = helper.make_graph(nodes,'bounded',[helper.make_tensor_value_info('x',TensorProto.FLOAT,['B',3])],
            [helper.make_tensor_value_info('y',TensorProto.FLOAT,['B',3])],constants)
        expected = np.cumsum(x,axis=0,dtype=np.float32) if op=='CumSum' else np.exp(x)
        yield f'bounded_{op.lower()}', (helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]),
            {'x':x},{'y':expected},{'dimension_bounds':{'B':{'min':1,'max':4}}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('build/tts-validation/cases.json'))
    args=parser.parse_args(); cases=[]
    with tempfile.TemporaryDirectory(prefix='volvox-tts-oracles-') as directory:
        root=Path(directory)
        for name,(model,inputs,expected,options) in fixtures():
            source=root/f'{name}.onnx'; onnx.save(model,source)
            package=root/name; package.mkdir()
            graph=export_onnx_model(str(source),str(package/'model.safetensors'),**options)
            # Exercise both schema forms through the public runtime.
            for node in graph['nodes']:
                if node['opType']=='CumSum' and not name.startswith('cumsum_float32_0_'):
                    for flag in ('exclusive','reverse'):
                        node['params'][flag]=bool(node['params'].get(flag,0))
            cases.append(dict(name=name,graph=graph,
                bitwise=name.startswith('expand_fill_'),
                weight_shards=[base64.b64encode((package/'model.safetensors').read_bytes()).decode()],
                inputs=[encoded(f'input{i}',v) for i,v in enumerate(inputs.values())],
                expected=[encoded(f'output{i}',v) for i,v in enumerate(expected.values())]))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'cases':cases},indent=2)+'\n')
    print(f'Wrote {len(cases)} independent oracles to {args.out}')

if __name__=='__main__': main()
