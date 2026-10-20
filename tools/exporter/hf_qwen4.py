"""Qwen3.8 Flash Next's Qwen4Exp graph, including HC, QSA and PLE.

Attention, recurrent scans and routed experts reuse the checked decoder
lowering. The family-specific transformation replaces its residual topology
with the official gated hyper-connections and inserts the lexical/indexed
branches. It produces the same common RuntimeIR, without executing a model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .hf_graph import GraphBuilder, positive_integer as _positive_integer


def _fail(message, code="VXHF_CONFIG"):
    raise ExporterError(Diagnostic(code, message, "hf-qwen4"))


def _splitmix(value):
    mask=(1<<64)-1
    value=(value+0x9E3779B97F4A7C15)&mask
    value=((value^(value>>30))*0xBF58476D1CE4E5B9)&mask
    value=((value^(value>>27))*0x94D049BB133111EB)&mask
    return (value^(value>>31))&mask


def _prime(number):
    return number>=2 and (number==2 or number%2!=0 and all(number%i for i in range(3,math.isqrt(number)+1,2)))


@dataclass(frozen=True)
class Qwen4DecoderConfig:
    decoder: object
    hc_count: int
    hc_lowrank: int
    ple_layers: tuple[int,...]
    ple_dim: int
    ple_kernel: int
    ngram_size: int
    ngram_heads: int
    ngram_base: int
    ngram_divisor: int
    ngram_parts: int
    seed: int
    eos: int
    index_heads: int
    index_dim: int
    index_budget: int
    index_rate: int
    linear_gate: str
    vision_config: Mapping|None=None
    family: str="qwen4_exp"

    def __getattr__(self,name):
        return getattr(self.decoder,name)

    @classmethod
    def from_hf(cls,source,*,max_context=None):
        from .hf_extended import ExtendedDecoderConfig
        outer=source.get("model_type")=="qwen4_exp" and "text_config" in source
        text=source.get("text_config") if outer else source
        if not isinstance(text,Mapping) or source.get("model_type") not in ("qwen4_exp","qwen4_exp_text"):
            _fail("source must identify Qwen4Exp text or multimodal architecture")
        if outer and text.get("model_type","qwen4_exp_text")!="qwen4_exp_text":
            _fail("outer and text_config model_type disagree")
        if source.get("architectures") not in (None,["Qwen4ExpForCausalLM"],["Qwen4ExpForConditionalGeneration"]):
            _fail("architectures must identify Qwen4ExpForCausalLM/ConditionalGeneration")
        if source.get("quantization_config") not in (None,{}):
            _fail("quantized source storage is deferred; use F32/F16/BF16")
        raw=dict(text)
        raw["model_type"]="qwen3_5_moe_text"
        raw.pop("architectures",None)
        if "layer_types" in raw:
            if any(kind not in ("linear_attention","indexed_attention","full_attention") for kind in raw["layer_types"]):
                _fail("Qwen4Exp layers must use linear or indexed attention")
            raw["layer_types"]=["full_attention" if kind=="indexed_attention" else kind for kind in raw["layer_types"]]
        normalize=text.get("norm_topk_prob",True)
        if not isinstance(normalize,bool):
            _fail("norm_topk_prob must be boolean")
        raw["norm_topk_prob"]=True
        base=replace(ExtendedDecoderConfig.from_hf(raw,max_context=max_context),normalize_routes=normalize)
        count=_positive_integer(text.get("hc_count",4),"hc_count")
        if count<=1:
            _fail("Qwen4Exp requires at least two hyper-connection streams")
        lowrank=_positive_integer(text.get("hc_lowrank",320),"hc_lowrank")
        ple=text.get("ple_layer_ids") or []
        if not isinstance(ple,list) or len(set(ple))!=len(ple) or any(not isinstance(i,int) or isinstance(i,bool) or not 1<=i<=base.layers or base.layer_types[i-1]!="linear_attention" for i in ple):
            _fail("ple_layer_ids must contain unique one-based linear-attention layer indices")
        ngram=_positive_integer(text.get("ngram_size",3),"ngram_size")
        heads=_positive_integer(text.get("heads_per_ngram",8),"heads_per_ngram")
        ple_dim=_positive_integer(text.get("ple_embed_dim",base.hidden_size),"ple_embed_dim")
        if ngram<2 or ple_dim%((ngram-1)*heads):
            _fail("PLE feature dimension must be divisible by all n-gram heads")
        seed=text.get("seed",1234)
        if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<1<<63:
            _fail("seed must be a nonnegative int64")
        eos=text.get("eos_token_id")
        eos=eos[0] if isinstance(eos,list) and eos else eos
        if not isinstance(eos,int) or isinstance(eos,bool) or not 0<=eos<base.vocab_size:
            _fail("eos_token_id must identify a token in the vocabulary")
        index=[]
        for field in ("indexer_n_heads","indexer_head_dim","indexer_budget","indexer_compress_ratio"):
            index.append(_positive_integer(text.get(field),field))
        if text.get("indexer_kv_heads",1)!=1 or index[1]<base.rotary_dim or index[2]%index[3]:
            _fail("QSA requires one key head, sufficient rotary dimensions, and budget divisible by compression ratio")
        linear_gate=text.get("output_gate_type") or text.get("hidden_act","silu")
        if linear_gate not in ("sigmoid","silu","swish"):
            _fail("recurrent output gate must be sigmoid or SiLU")
        vision=source.get("vision_config") if outer else None
        if vision is not None and not isinstance(vision,Mapping):
            _fail("vision_config must be an object")
        config=cls(base,count,lowrank,tuple(i-1 for i in sorted(ple)),ple_dim,
            _positive_integer(text.get("ple_conv_kernel_size",4),"ple_conv_kernel_size"),ngram,heads,
            _positive_integer(text.get("ngram_vocab_size_base",20000000),"ngram_vocab_size_base"),
            _positive_integer(text.get("make_ngram_vocab_size_divisible_by",128),"make_ngram_vocab_size_divisible_by"),
            _positive_integer(text.get("split_ngram_parts",512),"split_ngram_parts"),seed,eos,*index,linear_gate,vision)
        for layer in config.ple_layers:
            config.ngram_tables(layer)
        return config

    def ngram_tables(self,layer):
        index=self.ple_layers.index(layer)
        heads=(self.ngram_size-1)*self.ngram_heads
        prime=self.ngram_base-1
        sizes=[]
        for i in range((index+1)*heads):
            prime+=1
            while not _prime(prime):prime+=1
            if i>=index*heads:sizes.append(prime)
        offsets=np.cumsum([0]+sizes[:-1],dtype=np.int64)
        rows=math.ceil(sum(sizes)/self.ngram_divisor)*self.ngram_divisor
        if rows>2**31-1:
            _fail("n-gram vocabulary exceeds the engine's exact int32 index domain")
        half=max(1,((1<<63)-1)//max(self.vocab_size,1)//2)
        multipliers=np.asarray([2*(_splitmix((self.seed+10007*index+0x9E3779B97F4A7C15*(i+1))&((1<<64)-1))%half)+1 for i in range(self.ngram_size)],dtype=np.int64)
        return multipliers,np.asarray(sizes,np.int64),offsets,rows

    def weight_shapes(self):
        shapes=dict(self.decoder.weight_shapes())
        shapes.pop("model.norm.weight")
        h,hc=self.hidden_size,self.hidden_size*self.hc_count
        for layer,kind in enumerate(self.layer_types):
            p=f"model.layers.{layer}."
            shapes.pop(p+"input_layernorm.weight");shapes.pop(p+"post_attention_layernorm.weight")
            for label in ("attn_hyper_connection.","mlp_hyper_connection."):
                shapes[p+label+"hc_norm.weight"]=(hc,)
                shapes[p+label+"input_mix_weight_down.weight"]=(self.hc_lowrank,hc)
                shapes[p+label+"input_mix_weight_up.weight"]=(hc,self.hc_lowrank)
                shapes[p+label+"block_inject_weight.weight"]=(self.hc_count,hc)
            if kind!="linear_attention":
                prefix=p+"self_attn.indexer."
                shapes[prefix+"index_qk_proj.weight"]=((self.index_heads+1)*self.index_dim,h)
                shapes[prefix+"q_layernorm.weight"]=shapes[prefix+"k_layernorm.weight"]=(self.index_dim,)
            if layer in self.ple_layers:
                prefix=p+"ple."
                _,_,_,rows=self.ngram_tables(layer)
                shapes[prefix+"ple_embedding.ngram_embedding.weight"]=(rows,self.ple_dim//((self.ngram_size-1)*self.ngram_heads))
                shapes[prefix+"key_proj.weight"]=(hc,self.ple_dim)
                shapes[prefix+"value_proj.weight"]=(h,self.ple_dim)
                for label in ("norm_key","norm_query","norm_conv"):
                    shapes[prefix+label+".weight"]=(hc,)
                shapes[prefix+"conv1d.weight"]=(hc,1,self.ple_kernel)
        prefix="model.hyper_connection_mixer."
        shapes[prefix+"hc_norm.weight"]=(hc,)
        shapes[prefix+"input_mix_weight_down.weight"]=(self.hc_lowrank,hc)
        shapes[prefix+"input_mix_weight_up.weight"]=(hc,self.hc_lowrank)
        return shapes


def _weights(config,source):
    selected={}
    for name,value in source.items():
        if name.startswith("model.language_model."):
            name="model."+name[len("model.language_model."):]
        elif name.startswith("language_model.model."):
            name="model."+name[len("language_model.model."):]
        elif name.startswith("language_model.lm_head."):
            name="lm_head."+name[len("language_model.lm_head."):]
        elif config.vision_config is not None and name.startswith(("model.visual.","visual.")):
            continue
        elif name.startswith("mtp."):
            continue
        if name in selected:_fail(f"duplicate normalized tensor {name}","VXHF_WEIGHTS")
        selected[name]=value
    expected=config.weight_shapes()
    metadata={}
    shards={}
    for layer in config.ple_layers:
        p=f"model.layers.{layer}.ple.ple_embedding."
        tables=config.ngram_tables(layer)
        for suffix,table in zip(("layer_multipliers","ngram_heads_vocab_sizes","ngram_heads_offsets"),tables[:3]):
            name=p+suffix
            if name in selected:
                value=np.asarray(selected.pop(name))
                if value.dtype not in (np.dtype("int32"),np.dtype("int64")) or not np.array_equal(value,table):
                    _fail(f"n-gram metadata {name} disagrees with the deterministic configuration","VXHF_WEIGHTS")
            metadata[name]=table
        bank=p+"ngram_embedding."
        if bank+"weight" not in selected:
            expected.pop(bank+"weight")
            parts=[]
            rows=tables[3]
            if config.ngram_parts>rows:_fail("n-gram shard count exceeds vocabulary rows","VXHF_WEIGHTS")
            offset=0
            for shard in range(config.ngram_parts):
                count=rows//config.ngram_parts+int(shard<rows%config.ngram_parts)
                name=bank+f"shard_{shard}.weight"
                expected[name]=(count,config.ple_dim//((config.ngram_size-1)*config.ngram_heads))
                parts.append((name,offset));offset+=count
            shards[layer]=parts
        else:shards[layer]=[(bank+"weight",0)]
    optional={"lm_head.weight"} if config.tied_embeddings else set()
    if set(selected)-set(expected)-optional or set(expected)-set(selected):
        _fail(f"Qwen4 tensor mismatch: missing={sorted(set(expected)-set(selected))}, unexpected={sorted(set(selected)-set(expected)-optional)}","VXHF_WEIGHTS")
    result={}
    for name,shape in expected.items():
        value=np.asarray(selected[name])
        if value.dtype not in (np.dtype("float32"),np.dtype("float16")) or value.shape!=shape or not np.isfinite(value).all():
            _fail(f"tensor {name} must contain finite floating shape {shape}","VXHF_WEIGHTS")
        result[name]=np.ascontiguousarray(value,dtype=np.float32)
    if config.tied_embeddings and "lm_head.weight" in selected and not np.array_equal(selected["lm_head.weight"],result["model.embed_tokens.weight"]):
        _fail("tied lm_head.weight must equal embeddings","VXHF_WEIGHTS")
    return result,metadata,shards


def build_qwen4_decoder_graph(config,source_weights,*,output_name="logits",capture_kv=False,
        compact_kv=True,cached=False,generate=False,image_features_count=None):
    from .hf_extended import build_extended_decoder_graph, _finish_extended
    if not isinstance(cached, bool) or not isinstance(generate, bool):
        _fail("cached and generate must be boolean")
    if cached and capture_kv:
        _fail("cached execution requires private retained recurrent, KV and history tensors")
    weights,metadata,shards=_weights(config,source_weights)
    core={name:value for name,value in weights.items() if name in config.decoder.weight_shapes()}
    core["model.norm.weight"]=np.zeros((config.hidden_size,),np.float32)
    for layer in range(config.layers):
        for suffix in ("input_layernorm.weight","post_attention_layernorm.weight"):
            core[f"model.layers.{layer}."+suffix]=np.zeros((config.hidden_size,),np.float32)
    graph,initializers=build_extended_decoder_graph(config.decoder,core,output_name=output_name,
        capture_kv=capture_kv,compact_kv=True,cached=cached,image_features_count=image_features_count)
    # The core supplies retained recurrent state. Generation belongs to the
    # completed hyper-connection graph, whose logits replace the core logits.
    if cached:
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] != "generated"]
        graph["outputs"] = [name for name in graph["outputs"] if name != "generated"]
    b=GraphBuilder(config.max_context,graph["outputs"],graph["inputs"])
    b.weights.update(initializers)
    for name,value in weights.items():
        if name not in b.weights:b.add(name,value)
    s,h,c="S",config.hidden_size,config.hc_count
    hc=h*c
    b.add("hf.qwen4.hc_divisor",np.array([c],np.float32))
    b.add("hf.qwen4.unit",np.ones((h,),np.float32))
    b.add("hf.qwen4.two",np.array([2.],np.float32))
    b.add("hf.qwen4.root_hidden",np.array([math.sqrt(h)],np.float32))
    b.add("hf.qwen4.eos",np.array([config.eos],np.int32))

    def expand(name,value,shape,dtype="float32"):
        return b.tensor(name,"Expand",{"input":value},shape,{"shape":shape},dtype)

    def grouped_norm(name,value,weight):
        groups=b.reshape(name+".groups",value,[1,s,c,h])
        normalized=b.norm(name+".norm",groups,"hf.qwen4.unit",[1,s,c,h],config.epsilon)
        normalized=b.reshape(name+".flat",normalized,[1,s,hc])
        derived=weight+".hf_one"
        if derived not in b.weights:b.add(derived,weights[weight]+np.float32(1))
        scale=expand(name+".scale",derived,[1,s,hc])
        return b.tensor(name,"Mul",{"a":normalized,"b":scale},[1,s,hc])

    def mix(name,value,prefix,combine=True):
        normalized=grouped_norm(name+".group_norm",value,prefix+"hc_norm.weight")
        down=b.linear(name+".mix_down",normalized,prefix+"input_mix_weight_down.weight",config.hc_lowrank)
        down=b.tensor(name+".divide_streams","Div",{"a":down,"b":"hf.qwen4.hc_divisor"},[1,s,config.hc_lowrank])
        down=b.tensor(name+".mix_silu","SiLU",{"input":down},[1,s,config.hc_lowrank])
        up=b.linear(name+".mix_up",down,prefix+"input_mix_weight_up.weight",hc)
        up=b.tensor(name+".mix_sigmoid","Sigmoid",{"input":up},[1,s,hc])
        product=b.tensor(name+".mix_product","Mul",{"a":up,"b":normalized},[1,s,hc])
        product=b.reshape(name+".mix_groups",product,[1,s,c,h])
        product=b.tensor(name+".mix_last_stream","Transpose",{"input":product},[1,s,h,c],{"perm":[0,1,3,2]})
        mixed=b.tensor(name,"ReduceMean",{"input":product},[1,s,h],{"axis":-1,"keepdims":False})
        if not combine:return mixed,None
        gate=b.linear(name+".inject_projection",normalized,prefix+"block_inject_weight.weight",c)
        gate=b.tensor(name+".inject_divide","Div",{"a":gate,"b":"hf.qwen4.hc_divisor"},[1,s,c])
        gate=b.tensor(name+".inject_sigmoid","Sigmoid",{"input":gate},[1,s,c])
        two=expand(name+".inject_two","hf.qwen4.two",[1,s,c])
        gate=b.tensor(name+".inject_gate","Mul",{"a":gate,"b":two},[1,s,c])
        return mixed,gate

    def inject(name,value,block,gate):
        rows=b.reshape(name+".block_rows",block,[1,s,1,h])
        rows=expand(name+".block_expand",rows,[1,s,c,h])
        gate=b.reshape(name+".gate_rows",gate,[1,s,c,1])
        gate=expand(name+".gate_expand",gate,[1,s,c,h])
        product=b.tensor(name+".product","Mul",{"a":rows,"b":gate},[1,s,c,h])
        product=b.reshape(name+".flat",product,[1,s,hc])
        return b.tensor(name,"Add",{"a":value,"b":product},[1,s,hc])

    def ple(name,value,layer):
        p=f"model.layers.{layer}.ple."
        bank=p+"ple_embedding."
        multipliers,sizes,offsets,rows=config.ngram_tables(layer)
        words=np.stack((multipliers.astype(np.uint64)&np.uint64(0xffffffff),multipliers.astype(np.uint64)>>np.uint64(32)),axis=1).astype(np.uint32).view(np.int32)
        b.add(bank+"hf_multipliers",words);b.add(bank+"hf_sizes",sizes.astype(np.int32));b.add(bank+"hf_offsets",offsets.astype(np.int32))
        eos=expand(name+".eos","hf.qwen4.eos",[1,s],"int32")
        tokens=b.tensor(name+".tokens","Where",{"condition":"attention_mask","a":"token_ids","b":eos},[1,s],{},"int32")
        heads=(config.ngram_size-1)*config.ngram_heads
        ids=b.tensor(name+".ids","NGramIds",{"token_ids":tokens,"multipliers":bank+"hf_multipliers","vocab_sizes":bank+"hf_sizes","offsets":bank+"hf_offsets"},
            [1,s,heads],{"ngram_size":config.ngram_size,"heads_per_ngram":config.ngram_heads,"eos_token_id":config.eos,"vocab_size":config.vocab_size},"int32")
        embedded=None
        for shard,(weight,offset) in enumerate(shards[layer]):
            part=b.tensor(name+".embedding."+str(shard),"EmbeddingShard",{"input":ids,"weight":weight},
                [1,s,heads,config.ple_dim//heads],{"vocab_offset":offset,"vocab_size":rows})
            embedded=part if embedded is None else b.tensor(name+".embedding_sum."+str(shard),"Add",{"a":embedded,"b":part},[1,s,heads,config.ple_dim//heads])
        embedded=b.reshape(name+".features",embedded,[1,s,config.ple_dim])
        key=b.linear(name+".key",embedded,p+"key_proj.weight",hc)
        key=grouped_norm(name+".key_norm",key,p+"norm_key.weight")
        query=grouped_norm(name+".query_norm",value,p+"norm_query.weight")
        product=b.tensor(name+".product","Mul",{"a":key,"b":query},[1,s,hc])
        product=b.reshape(name+".product_groups",product,[1,s,c,h])
        gate=b.tensor(name+".dot","ReduceSum",{"input":product},[1,s,c,1],{"axis":-1,"keepdims":True})
        gate=b.tensor(name+".scale","Div",{"a":gate,"b":"hf.qwen4.root_hidden"},[1,s,c,1])
        gate=b.tensor(name+".signed_root","SignedSqrt",{"input":gate},[1,s,c,1],{"eps":1e-6})
        gate=b.tensor(name+".sigmoid","Sigmoid",{"input":gate},[1,s,c,1])
        gate=expand(name+".gate_expand",gate,[1,s,c,h])
        projected=b.linear(name+".value",embedded,p+"value_proj.weight",h)
        projected=b.reshape(name+".value_rows",projected,[1,s,1,h])
        projected=expand(name+".value_expand",projected,[1,s,c,h])
        gated=b.tensor(name+".gated","Mul",{"a":projected,"b":gate},[1,s,c,h])
        gated=b.reshape(name+".gated_flat",gated,[1,s,hc])
        normalized=grouped_norm(name+".conv_norm",gated,p+"norm_conv.weight")
        mask=b.reshape(name+".mask_rows","attention_mask",[1,s,1],"int32")
        mask=expand(name+".mask_expand",mask,[1,s,hc],"int32")
        zero=expand(name+".zero_expand","hf.zero",[1,s,hc])
        gated=b.tensor(name+".masked_gated","Where",{"condition":mask,"a":gated,"b":zero},[1,s,hc])
        normalized=b.tensor(name+".masked_norm","Where",{"condition":mask,"a":normalized,"b":zero},[1,s,hc])
        normalized=b.reshape(name+".conv_input",normalized,[1,1,s,hc])
        kernel=b.add(p+"hf_conv.weight",weights[p+"conv1d.weight"][:,0,:].T[None,:,:,None])
        convolved=b.tensor(name+".conv","Conv2D",{"input":normalized,"weight":kernel},[1,1,s,hc],
            {"groups":hc,"pads":[0,(config.ple_kernel-1)*config.ngram_size,0,0],"dilation":[1,config.ngram_size],"data_layout":"NHWC","weight_layout":"HWCM"})
        convolved=b.reshape(name+".conv_rows",convolved,[1,s,hc])
        convolved=b.tensor(name+".conv_silu","SiLU",{"input":convolved},[1,s,hc])
        output=b.tensor(name+".output","Add",{"a":gated,"b":convolved},[1,s,hc])
        return b.tensor(name,"Add",{"a":value,"b":output},[1,s,hc])

    stream=None
    residual=None
    injection_gate=None
    for node in graph["nodes"]:
        name=node["id"]
        if name=="embedding" or name=="image.embedding":
            b.graph["nodes"].append(node)
            # Images replace token features before their repetition into streams.
            if name=="embedding" and image_features_count is not None:continue
            rows=b.reshape("hf.qwen4.embedding_rows",name,[1,s,1,h])
            rows=expand("hf.qwen4.embedding_expand",rows,[1,s,c,h])
            stream=b.reshape("hf.qwen4.embedding_streams",rows,[1,s,hc])
        elif name.endswith(".attention_norm"):
            layer=int(name.split(".")[1]);prefix=f"model.layers.{layer}."
            if layer in config.ple_layers:stream=ple(f"layer.{layer}.ple",stream,layer)
            residual=stream
            _,injection_gate=mix(name,stream,prefix+"attn_hyper_connection.")
        elif name.endswith(".attention_residual"):
            stream=inject(name,residual,node["inputs"]["b"],injection_gate)
        elif name.endswith(".mlp_norm"):
            layer=int(name.split(".")[1]);residual=stream
            _,injection_gate=mix(name,stream,f"model.layers.{layer}.mlp_hyper_connection.")
        elif name.endswith(".mlp_residual"):
            stream=inject(name,residual,node["inputs"]["b"],injection_gate)
        elif name=="final_norm":
            mix(name,stream,"model.hyper_connection_mixer.",False)
        elif node["opType"]=="SparseAttention":
            layer=int(name.split(".")[1]);prefix=f"model.layers.{layer}.self_attn.indexer."
            hidden=f"layer.{layer}.attention_norm"
            packed=weights[prefix+"index_qk_proj.weight"]
            qw=b.add(prefix+"hf_q.weight",packed[:config.index_heads*config.index_dim])
            kw=b.add(prefix+"hf_k.weight",packed[config.index_heads*config.index_dim:])
            q=b.linear(name+".index_q",hidden,qw,config.index_heads*config.index_dim)
            q=b.reshape(name+".index_q_heads",q,[1,s,config.index_heads,config.index_dim])
            qnorm=b.add(prefix+"hf_qnorm",weights[prefix+"q_layernorm.weight"]+np.float32(1))
            knorm=b.add(prefix+"hf_knorm",weights[prefix+"k_layernorm.weight"]+np.float32(1))
            q=b.norm(name+".index_q_norm",q,qnorm,[1,s,config.index_heads,config.index_dim],config.epsilon)
            q=b.tensor(name+".index_q_rope","RotaryEmbedding",{"input":q,"cos":"hf.rope.cos.rows","sin":"hf.rope.sin.rows"},[1,s,config.index_heads,config.index_dim],{"rotary_dim":config.rotary_dim,"interleaved":False})
            k=b.linear(name+".index_k",hidden,kw,config.index_dim)
            mask=b.tensor(name+".indexer","QSAIndexer",{"q":q,"k":k,"weight":knorm,"cos":"hf.rope.cos.rows","sin":"hf.rope.sin.rows","mask":"attention_mask"},
                [1,s,s],{"compress_rate":config.index_rate,"top_k":config.index_budget//config.index_rate,"rotary_dim":config.rotary_dim,"eps":config.epsilon,"causal":True},"int32")
            node=dict(node);node["inputs"]={**node["inputs"],"mask":mask};b.graph["nodes"].append(node)
        elif name.endswith(".z_silu") and config.linear_gate=="sigmoid":
            node=dict(node);node["opType"]="Sigmoid";b.graph["nodes"].append(node)
        else:b.graph["nodes"].append(node)
    return _finish_extended(b,config,output_name,generate,cached=cached)
