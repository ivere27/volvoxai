"""Bounded vision graphs for Qwen3.5-compatible multimodal checkpoints.

The input consists of the flattened, normalized patches produced by the model's
image/video processor. Fixed exported grids determine attention boundaries,
learned-position interpolation and the spatial merger; no image decoder or
alternate inference runtime is embedded in the exporter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .hf_graph import (GraphBuilder, positive_float as _positive_float, positive_integer as _positive_integer,
    read_checkpoint_storage, write_hf_package)


def _fail(message):
    raise ExporterError(Diagnostic("VXHF_VISION", message, "hf-vision"))


def export_hf_vision(directory, out_path, *, grid_thw=None, tiles=None,
        output_names=None, report_callback=None, weight_precision=None, group_size=128):
    """Publish a fixed-grid image encoder/projector as its own model package.

    Its image_features output feeds the text decoder's identically named input.
    Normalization, tile selection and token-to-feature routing remain explicit
    application inputs produced by the checkpoint's official processor.
    """
    from .hf_extended import ExtendedDecoderConfig
    source, source_weights, source_dtypes = read_checkpoint_storage(directory,
        config_validator=ExtendedDecoderConfig.from_hf)
    config = ExtendedDecoderConfig.from_hf(source)
    if config.vision_config is None:
        _fail("checkpoint does not declare a vision encoder")
    if output_names is not None and len(output_names) != 1:
        _fail("HF vision export has exactly one image features output")
    output = output_names[0] if output_names else "image_features"
    if config.family == "llama4":
        if grid_thw is not None or tiles is None:
            _fail("Llama4 vision export requires --hf-vision-tiles only")
        graph, weights = build_llama4_vision_graph(Llama4VisionConfig.from_hf(config.vision_config),
            source_weights, tiles=tiles, text_hidden_size=config.hidden_size, output_name=output)
        layout = "normalized-NCHW-tiles"
    else:
        if config.family not in {"qwen3_5", "qwen3_5_moe", "qwen4_exp"} or grid_thw is None or tiles is not None:
            _fail("Qwen multimodal vision export requires --hf-vision-grid only")
        graph, weights = build_qwen_vision_graph(QwenVisionConfig.from_hf(config.vision_config),
            source_weights, grid_thw=grid_thw, output_name=output)
        layout = "normalized-flattened-temporal-patches"
    graph = write_hf_package(out_path, graph, weights, weight_precision=weight_precision, group_size=group_size)
    if report_callback:
        report_callback({"features": {"hf_vision": [f"family:{config.family}", "scope:image-encoder-and-projector",
            "execution:F32", f"input:{layout}", "routing:explicit-image-feature-IDs-in-separate-decoder"],
            "source_storage": sorted(set(source_dtypes.values()))}})
    return graph


@dataclass(frozen=True)
class QwenVisionConfig:
    hidden: int
    intermediate: int
    heads: int
    depth: int
    channels: int
    patch: int
    temporal_patch: int
    merge: int
    positions: int
    output: int
    theta: float
    activation: str

    @classmethod
    def from_hf(cls, raw):
        if not isinstance(raw, Mapping):
            _fail("vision_config must be an object")
        if raw.get("model_type") not in (None, "qwen3_5", "qwen3_5_moe", "qwen4_exp",
                "qwen3_5_vision", "qwen3_5_moe_vision", "qwen4_exp_vision"):
            _fail("vision_config does not identify the Qwen3.5-compatible patch encoder")
        if raw.get("deepstack_visual_indexes") not in (None, []):
            _fail("deepstack vision features require a distinct decoder connection")
        if raw.get("quantization_config") not in (None, {}):
            _fail("quantized vision source storage is unsupported")
        hidden = _positive_integer(raw.get("hidden_size", 1152), "vision.hidden_size")
        heads = _positive_integer(raw.get("num_heads", 16), "vision.num_heads")
        if hidden % heads or (hidden // heads) % 4:
            _fail("vision head dimension must be divisible by four for axial RoPE")
        positions = _positive_integer(raw.get("num_position_embeddings", 2304), "vision.num_position_embeddings")
        if math.isqrt(positions) ** 2 != positions:
            _fail("learned vision positions must form a square grid")
        rope = raw.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.}
        if not isinstance(rope, Mapping) or set(rope) - {"rope_type", "rope_theta"} or rope.get("rope_type", "axial") != "axial":
            _fail("vision supports fixed axial RoPE only")
        activation = raw.get("hidden_act", "gelu_pytorch_tanh")
        if activation not in ("gelu_pytorch_tanh", "gelu", "gelu_new"):
            _fail("vision MLP activation must be GELU or tanh GELU")
        return cls(hidden, _positive_integer(raw.get("intermediate_size", 4304), "vision.intermediate_size"),
            heads, _positive_integer(raw.get("depth", 27), "vision.depth"),
            _positive_integer(raw.get("in_channels", 3), "vision.in_channels"),
            _positive_integer(raw.get("patch_size", 16), "vision.patch_size"),
            _positive_integer(raw.get("temporal_patch_size", 2), "vision.temporal_patch_size"),
            _positive_integer(raw.get("spatial_merge_size", 2), "vision.spatial_merge_size"), positions,
            _positive_integer(raw.get("out_hidden_size", 2048), "vision.out_hidden_size"),
            _positive_float(rope.get("rope_theta", 10000.), "vision.rope_theta"), activation)

    def weight_shapes(self):
        h, m = self.hidden, self.hidden * self.merge ** 2
        result = {"patch_embed.proj.weight": (h, self.channels, self.temporal_patch, self.patch, self.patch),
            "patch_embed.proj.bias": (h,), "pos_embed.weight": (self.positions, h),
            "merger.norm.weight": (h,), "merger.norm.bias": (h,),
            "merger.linear_fc1.weight": (m, m), "merger.linear_fc1.bias": (m,),
            "merger.linear_fc2.weight": (self.output, m), "merger.linear_fc2.bias": (self.output,)}
        for layer in range(self.depth):
            p = f"blocks.{layer}."
            for norm in ("norm1", "norm2"):
                result[p + norm + ".weight"] = result[p + norm + ".bias"] = (h,)
            for name, shape in (("attn.qkv", (h * 3, h)), ("attn.proj", (h, h)),
                    ("mlp.linear_fc1", (self.intermediate, h)), ("mlp.linear_fc2", (h, self.intermediate))):
                result[p + name + ".weight"], result[p + name + ".bias"] = shape, (shape[0],)
        return result


def _vision_weights(config, source):
    expected = config.weight_shapes()
    selected = {}
    for name, value in source.items():
        if name.startswith("model.visual."):
            name = name[len("model.visual."):]
        elif name.startswith("visual."):
            name = name[len("visual."):]
        elif name not in expected:
            # A full multimodal checkpoint contains explicitly separate text
            # and optional speculative subgraphs; only the named vision bank
            # is selected here. Unknown vision names still reach exact refusal.
            if name.startswith(("model.language_model.", "model.layers.", "model.embed_tokens.", "model.norm.", "lm_head.", "mtp.")):
                continue
        if name in selected:
            _fail(f"duplicate vision tensor {name}")
        selected[name] = value
    if set(selected) != set(expected):
        _fail(f"vision tensor mismatch: missing={sorted(set(expected)-set(selected))}, unexpected={sorted(set(selected)-set(expected))}")
    result = {}
    for name, shape in expected.items():
        value = np.asarray(selected[name])
        if value.dtype not in (np.dtype("float32"), np.dtype("float16")) or value.shape != shape or not np.isfinite(value).all():
            _fail(f"vision tensor {name} must contain finite floating shape {shape}")
        result[name] = np.ascontiguousarray(value, dtype=np.float32)
    return result


def _fixed_grid(config, grid_thw):
    grid = np.asarray(grid_thw)
    if grid.dtype.kind not in "iu" or grid.ndim != 2 or grid.shape[1] != 3 or not len(grid) or (grid <= 0).any():
        _fail("grid_thw must contain positive integer [images,3] temporal/height/width dimensions")
    if (grid[:, 1:] % config.merge).any():
        _fail("vision height and width grids must be divisible by spatial_merge_size")
    count = sum(int(t) * int(h) * int(w) for t, h, w in grid)
    if count > 2 ** 31 - 1 or count * count * 4 > 256 << 20:
        _fail("fixed-grid attention mask exceeds 256 MiB; export smaller image/video grids")
    coordinates, segments = [], []
    for t, height, width in grid:
        height, width = int(height), int(width)
        row, column = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        block = (height // config.merge, config.merge, width // config.merge, config.merge)
        row = row.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
        column = column.reshape(block).transpose(0, 2, 1, 3).reshape(-1)
        for _ in range(int(t)):
            coordinates.append(np.stack((row, column), axis=1))
            segments.extend([len(coordinates) - 1] * (height * width))
    return grid, np.concatenate(coordinates), np.asarray(segments, np.int32)


def build_qwen_vision_graph(config: QwenVisionConfig, source_weights, *, grid_thw, output_name="image_features"):
    weights = _vision_weights(config, source_weights)
    grid, coords, segments = _fixed_grid(config, grid_thw)
    n, h, d = len(coords), config.hidden, config.hidden // config.heads
    features = config.channels * config.temporal_patch * config.patch ** 2
    b = GraphBuilder(1, [output_name], {"pixel_patches": {"shape": [n, features], "dtype": "float32"}})
    b.graph["dimensions"] = {}
    b.weights.update(weights)
    b.add("hf.vision.patch.weight", weights["patch_embed.proj.weight"].reshape(h, features))
    hidden = b.linear("vision.patch", "pixel_patches", "hf.vision.patch.weight", h,
        "patch_embed.proj.bias", [n, h])
    # Interpolate the learned square grid at export time. It is immutable and
    # image-size dependent, so folding this linear initializer is exact.
    side = math.isqrt(config.positions)
    positions, offset = [], 0
    for t, height, width in grid:
        height, width = int(height), int(width)
        image_coords = coords[offset:offset + height * width]
        y = np.linspace(0, side - 1, height, dtype=np.float32)[image_coords[:, 0]]
        x = np.linspace(0, side - 1, width, dtype=np.float32)[image_coords[:, 1]]
        y0, x0 = np.floor(y).astype(np.int32), np.floor(x).astype(np.int32)
        y1, x1 = np.minimum(y0 + 1, side - 1), np.minimum(x0 + 1, side - 1)
        wy, wx = (y - y0).astype(np.float32), (x - x0).astype(np.float32)
        value = sum(weights["pos_embed.weight"][yy * side + xx] * factor[:, None] for yy, xx, factor in (
            (y0, x0, (1-wy)*(1-wx)), (y0, x1, (1-wy)*wx), (y1, x0, wy*(1-wx)), (y1, x1, wy*wx)))
        positions.extend([value] * int(t))
        offset += int(t) * height * width
    b.add("hf.vision.position", np.concatenate(positions).astype(np.float32))
    hidden = b.tensor("vision.positioned", "Add", {"a": hidden, "b": "hf.vision.position"}, [n, h])
    exponent = np.arange(0, d // 2, 2, dtype=np.float32) / np.float32(d // 2)
    inverse = np.float32(1) / np.power(np.float32(config.theta), exponent)
    phase = (coords.astype(np.float32)[:, :, None] * inverse).reshape(n, d // 2)
    phase = np.concatenate((phase, phase), axis=1)[None]
    for function in ("cos", "sin"):
        b.add("hf.vision." + function, getattr(np, function)(phase).astype(np.float32))
    b.add("hf.vision.mask", (segments[:, None] == segments[None, :])[None].astype(np.int32))
    hidden = b.reshape("vision.rows", hidden, [1, n, h])

    def norm(name, value, prefix):
        return b.tensor(name, "LayerNorm", {"input": value, "weight": prefix + ".weight", "bias": prefix + ".bias"}, [1, n, h], {"eps": 1e-6})

    for layer in range(config.depth):
        p, name = f"blocks.{layer}.", f"vision.layer.{layer}"
        normalized = norm(name + ".attention_norm", hidden, p + "norm1")
        qkv = []
        for index, label in enumerate(("q", "k", "v")):
            weight = b.add(p + "hf_" + label + ".weight", weights[p + "attn.qkv.weight"][index*h:(index+1)*h])
            bias = b.add(p + "hf_" + label + ".bias", weights[p + "attn.qkv.bias"][index*h:(index+1)*h])
            projected = b.linear(name + "." + label, normalized, weight, h, bias, [1, n, h])
            projected = b.reshape(name + "." + label + "_heads", projected, [1, n, config.heads, d])
            if label != "v":
                projected = b.tensor(name + "." + label + "_rope", "RotaryEmbedding", {"input": projected,
                    "cos": "hf.vision.cos", "sin": "hf.vision.sin"}, [1, n, config.heads, d], {"rotary_dim": d, "interleaved": False})
            qkv.append(projected)
        attention = b.tensor(name + ".attention", "SparseAttention", {"q": qkv[0], "k": qkv[1], "v": qkv[2], "mask": "hf.vision.mask"},
            [1, n, config.heads, d], {"causal": False, "scale": 1/math.sqrt(d)})
        attention = b.reshape(name + ".attention_flat", attention, [1, n, h])
        projected = b.linear(name + ".attention_output", attention, p + "attn.proj.weight", h, p + "attn.proj.bias", [1, n, h])
        hidden = b.tensor(name + ".attention_residual", "Add", {"a": hidden, "b": projected}, [1, n, h])
        normalized = norm(name + ".mlp_norm", hidden, p + "norm2")
        up = b.linear(name + ".up", normalized, p + "mlp.linear_fc1.weight", config.intermediate,
            p + "mlp.linear_fc1.bias", [1, n, config.intermediate])
        up = b.tensor(name + ".gelu", "GELU", {"input": up}, [1, n, config.intermediate],
            {"approximate": "none" if config.activation == "gelu" else "tanh"})
        down = b.linear(name + ".down", up, p + "mlp.linear_fc2.weight", h, p + "mlp.linear_fc2.bias", [1, n, h])
        hidden = b.tensor(name + ".mlp_residual", "Add", {"a": hidden, "b": down}, [1, n, h])
    normalized = norm("vision.merger_norm", hidden, "merger.norm")
    count, merged = n // config.merge ** 2, h * config.merge ** 2
    normalized = b.reshape("vision.merger_rows", normalized, [count, merged])
    up = b.linear("vision.merger_up", normalized, "merger.linear_fc1.weight", merged, "merger.linear_fc1.bias", [count, merged])
    up = b.tensor("vision.merger_gelu", "GELU", {"input": up}, [count, merged], {"approximate": "none"})
    b.linear(output_name, up, "merger.linear_fc2.weight", config.output, "merger.linear_fc2.bias", [count, config.output])
    return b.finish()


@dataclass(frozen=True)
class Llama4VisionConfig:
    hidden: int
    intermediate: int
    heads: int
    depth: int
    channels: int
    image: int
    patch: int
    shuffle: int
    projector: int
    output: int
    theta: float

    @classmethod
    def from_hf(cls, raw):
        if not isinstance(raw, Mapping) or raw.get("model_type") not in (None, "llama4_vision_model"):
            _fail("vision_config must identify Llama4VisionModel")
        if raw.get("quantization_config") not in (None, {}):
            _fail("quantized Llama4 vision source storage is unsupported")
        if raw.get("vision_feature_select_strategy", "default") != "default":
            _fail("Llama4 vision exports the final patch features only")
        hidden = _positive_integer(raw.get("hidden_size", 768), "vision.hidden_size")
        heads = _positive_integer(raw.get("num_attention_heads", 16), "vision.num_attention_heads")
        image = _positive_integer(raw.get("image_size", 448), "vision.image_size")
        patch = _positive_integer(raw.get("patch_size", 14), "vision.patch_size")
        ratio = _positive_float(raw.get("pixel_shuffle_ratio", .5), "vision.pixel_shuffle_ratio")
        shuffle = int(round(1 / ratio))
        if shuffle <= 0 or ratio != 1 / shuffle or image % patch or (image // patch) % shuffle:
            _fail("Llama4 pixel shuffle requires an integer reciprocal ratio dividing the patch grid")
        if hidden % heads or (hidden // heads) % 4:
            _fail("Llama4 axial rotary head dimension must be divisible by four")
        intermediate = _positive_integer(raw.get("intermediate_size", 5632), "vision.intermediate_size")
        projector = _positive_integer(raw.get("projector_input_dim", 4096), "vision.projector_input_dim")
        output = _positive_integer(raw.get("projector_output_dim", 4096), "vision.projector_output_dim")
        if intermediate != hidden * shuffle ** 2 or projector != output or raw.get("vision_output_dim", output) != output:
            _fail("Llama4 pixel-shuffled feature/adapter/projector dimensions must agree")
        # HF's encoder uses LayerNorm's fixed 1e-5 and exact GELU; norm_eps and
        # hidden_act are descriptive fields rather than alternate implementations.
        rope = raw.get("rope_parameters") or {"rope_type": "default", "rope_theta": 10000.}
        if not isinstance(rope, Mapping) or set(rope) - {"rope_type", "rope_theta"} or rope.get("rope_type", "default") != "default":
            _fail("Llama4 vision supports fixed default axial frequencies only")
        return cls(hidden, intermediate, heads,
            _positive_integer(raw.get("num_hidden_layers", 34), "vision.num_hidden_layers"),
            _positive_integer(raw.get("num_channels", 3), "vision.num_channels"), image, patch,
            shuffle, projector, output, _positive_float(rope.get("rope_theta", 10000.), "vision.rope_theta"))

    def weight_shapes(self):
        h, n = self.hidden, (self.image // self.patch) ** 2 + 1
        result = {"patch_embedding.linear.weight": (h, self.channels * self.patch ** 2),
            "class_embedding": (h,), "positional_embedding_vlm": (n, h),
            "vision_adapter.mlp.fc1.weight": (self.projector, self.intermediate),
            "vision_adapter.mlp.fc2.weight": (self.output, self.output)}
        for norm in ("layernorm_pre", "layernorm_post"):
            result[norm + ".weight"] = result[norm + ".bias"] = (h,)
        for layer in range(self.depth):
            p = f"model.layers.{layer}."
            for norm in ("input_layernorm", "post_attention_layernorm"):
                result[p + norm + ".weight"] = result[p + norm + ".bias"] = (h,)
            for name, shape in (("self_attn.q_proj", (h,h)), ("self_attn.k_proj", (h,h)),
                    ("self_attn.v_proj", (h,h)), ("self_attn.o_proj", (h,h)),
                    ("mlp.fc1", (self.intermediate,h)), ("mlp.fc2", (h,self.intermediate))):
                result[p + name + ".weight"], result[p + name + ".bias"] = shape, (shape[0],)
        return result


def build_llama4_vision_graph(config: Llama4VisionConfig, source_weights, *, tiles=1,
        text_hidden_size=None, output_name="image_features"):
    tiles = _positive_integer(tiles, "vision.tiles")
    expected = config.weight_shapes()
    selected = {}
    projector_weight = None
    for original, value in source_weights.items():
        if original in ("model.multi_modal_projector.linear_1.weight", "multi_modal_projector.linear_1.weight"):
            if projector_weight is not None:
                _fail("duplicate multimodal projector weight")
            projector_weight = value
            continue
        if original.startswith("model.vision_model."):
            name = original[len("model.vision_model."):]
        elif original.startswith("vision_model."):
            name = original[len("vision_model."):]
        elif original in expected:
            name = original
        elif original.startswith(("model.language_model.", "language_model.", "lm_head.")):
            continue
        else:
            _fail(f"unexpected Llama4 vision tensor {original}")
        if name in selected:
            _fail(f"duplicate Llama4 vision tensor {name}")
        selected[name] = value
    if set(selected) != set(expected):
        _fail(f"Llama4 vision tensor mismatch: missing={sorted(set(expected)-set(selected))}, unexpected={sorted(set(selected)-set(expected))}")
    weights = {}
    for name, shape in expected.items():
        value = np.asarray(selected[name])
        if value.dtype not in (np.dtype("float32"),np.dtype("float16")) or value.shape != shape or not np.isfinite(value).all():
            _fail(f"Llama4 vision tensor {name} must contain finite floating shape {shape}")
        weights[name] = np.ascontiguousarray(value, dtype=np.float32)
    h, grid = config.hidden, config.image // config.patch
    n, d = grid ** 2 + 1, h // config.heads
    if tiles * n * h > 2 ** 31 - 1:
        _fail("Llama4 tile graph exceeds int32 tensor geometry")
    b = GraphBuilder(1, [output_name], {"pixel_values": {"shape": [tiles,config.channels,config.image,config.image], "dtype":"float32"}})
    b.graph["dimensions"] = {}
    b.weights.update(weights)
    kernel = weights["patch_embedding.linear.weight"].reshape(h,config.channels,config.patch,config.patch).transpose(2,3,1,0)
    b.add("hf.llama4.patch.weight", kernel)
    pixels = b.tensor("vision.pixels_nhwc", "Transpose", {"input":"pixel_values"}, [tiles,config.image,config.image,config.channels], {"perm":[0,2,3,1]})
    hidden = b.tensor("vision.patch", "Conv2D", {"input":pixels,"weight":"hf.llama4.patch.weight"},
        [tiles,grid,grid,h], {"stride":[config.patch,config.patch],"data_layout":"NHWC","weight_layout":"HWIO"})
    hidden = b.reshape("vision.patch_rows",hidden,[tiles,n-1,h])
    cls = b.tensor("vision.cls", "Expand", {"input":"class_embedding"}, [tiles,1,h], {"shape":[tiles,1,h]})
    hidden = b.tensor("vision.with_cls","Concat",{"input0":hidden,"input1":cls},[tiles,n,h],{"axis":1})
    positions = b.tensor("vision.position_expand","Expand",{"input":"positional_embedding_vlm"},[tiles,n,h],{"shape":[tiles,n,h]})
    hidden = b.tensor("vision.positioned","Add",{"a":hidden,"b":positions},[tiles,n,h])

    def norm(name,value,prefix,shape=[tiles,n,h]):
        return b.tensor(name,"LayerNorm",{"input":value,"weight":prefix+".weight","bias":prefix+".bias"},shape,{"eps":1e-5})

    hidden = norm("vision.pre_norm",hidden,"layernorm_pre")
    indices = np.arange(n-1,dtype=np.int32)
    x,y = (indices%grid+1).astype(np.float32),(indices//grid+1).astype(np.float32)
    inverse = np.float32(1)/np.power(np.float32(config.theta),np.arange(0,d//2,2,dtype=np.float32)/np.float32(d//2))
    phase = np.concatenate((x[:,None]*inverse,y[:,None]*inverse),axis=1)
    phase = np.concatenate((phase,np.zeros((1,d//2),np.float32)),axis=0)[None]
    for function in ("cos","sin"):
        b.add("hf.llama4."+function,np.tile(getattr(np,function)(phase),(tiles,1,1)).astype(np.float32))
    for layer in range(config.depth):
        p,name=f"model.layers.{layer}.",f"vision.layer.{layer}"
        normalized=norm(name+".attention_norm",hidden,p+"input_layernorm")
        qkv=[]
        for label,proj in (("q","q_proj"),("k","k_proj"),("v","v_proj")):
            projected=b.linear(name+"."+label,normalized,p+"self_attn."+proj+".weight",h,p+"self_attn."+proj+".bias",[tiles,n,h])
            projected=b.reshape(name+"."+label+"_heads",projected,[tiles,n,config.heads,d])
            if label!="v":
                projected=b.tensor(name+"."+label+"_rope","RotaryEmbedding",{"input":projected,"cos":"hf.llama4.cos","sin":"hf.llama4.sin"},
                    [tiles,n,config.heads,d],{"rotary_dim":d,"interleaved":True})
            qkv.append(projected)
        attention=b.tensor(name+".attention","SparseAttention",dict(zip(("q","k","v"),qkv)),[tiles,n,config.heads,d],{"causal":False,"scale":1/math.sqrt(d)})
        attention=b.reshape(name+".attention_flat",attention,[tiles,n,h])
        projected=b.linear(name+".attention_output",attention,p+"self_attn.o_proj.weight",h,p+"self_attn.o_proj.bias",[tiles,n,h])
        hidden=b.tensor(name+".attention_residual","Add",{"a":hidden,"b":projected},[tiles,n,h])
        normalized=norm(name+".mlp_norm",hidden,p+"post_attention_layernorm")
        up=b.linear(name+".up",normalized,p+"mlp.fc1.weight",config.intermediate,p+"mlp.fc1.bias",[tiles,n,config.intermediate])
        up=b.tensor(name+".gelu","GELU",{"input":up},[tiles,n,config.intermediate],{"approximate":"none"})
        down=b.linear(name+".down",up,p+"mlp.fc2.weight",h,p+"mlp.fc2.bias",[tiles,n,h])
        hidden=b.tensor(name+".mlp_residual","Add",{"a":hidden,"b":down},[tiles,n,h])
    hidden=norm("vision.post_norm",hidden,"layernorm_post")
    b.add("hf.llama4.patches.indices",indices)
    hidden=b.tensor("vision.drop_cls","Gather",{"input":hidden,"indices":"hf.llama4.patches.indices"},[tiles,n-1,h],{"axis":1})
    # Reproduce the two row/column reshuffles in HF's pixel_shuffle exactly.
    k=config.shuffle
    hidden=b.reshape("vision.shuffle_width",hidden,[tiles,grid,grid//k,h*k])
    hidden=b.tensor("vision.shuffle_transpose1","Transpose",{"input":hidden},[tiles,grid//k,grid,h*k],{"perm":[0,2,1,3]})
    hidden=b.reshape("vision.shuffle_height",hidden,[tiles,grid//k,grid//k,h*k*k])
    hidden=b.tensor("vision.shuffle_transpose2","Transpose",{"input":hidden},[tiles,grid//k,grid//k,h*k*k],{"perm":[0,2,1,3]})
    merged=(grid//k)**2
    hidden=b.reshape("vision.adapter_rows",hidden,[tiles,merged,h*k*k])
    hidden=b.linear("vision.adapter_up",hidden,"vision_adapter.mlp.fc1.weight",config.projector,shape=[tiles,merged,config.projector])
    hidden=b.tensor("vision.adapter_gelu1","GELU",{"input":hidden},[tiles,merged,config.projector],{"approximate":"none"})
    hidden=b.linear("vision.adapter_down",hidden,"vision_adapter.mlp.fc2.weight",config.output,shape=[tiles,merged,config.output])
    hidden=b.tensor("vision.adapter_gelu2","GELU",{"input":hidden},[tiles,merged,config.output],{"approximate":"none"})
    hidden=b.reshape("vision.adapter_features",hidden,[tiles*merged,config.output])
    if text_hidden_size is not None:
        text_hidden_size=_positive_integer(text_hidden_size,"text_hidden_size")
        weight=np.asarray(projector_weight)
        if weight.dtype not in (np.dtype("float32"),np.dtype("float16")) or weight.shape!=(text_hidden_size,config.output) or not np.isfinite(weight).all():
            _fail("multimodal projector must contain the finite [text_hidden,vision_output] matrix")
        b.add("hf.llama4.projector.weight",np.asarray(weight,np.float32))
        b.linear(output_name,hidden,"hf.llama4.projector.weight",text_hidden_size,shape=[tiles*merged,text_hidden_size])
    else:
        b.tensor(output_name,"Identity",{"input":hidden},[tiles*merged,config.output])
    return b.finish()
