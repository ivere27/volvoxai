"""DeepSeek V4.1 text backbone: shared CSA2, delayed mHC and Engram.

This is the complete F32 text graph. It deliberately does not emulate the
reference's activation FP8/FP4 fake quantization, vision or DSpark drafts.
Checkpoint decoding is independent of that execution precision boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import math
import numpy as np

from .hf_deepseek import (_fail, _integer, _boolean, _rope_parameters, _rotary_tables,
    add_deepseek_moe, collapse_streams)
from .hf_graph import (GraphBuilder, positive_float, positive_integer,
    read_config, read_checkpoint_storage, write_hf_package)
from .runtime_names import is_runtime_graph_name


@dataclass(frozen=True)
class DeepSeekV41Config:
    vocab_size: int
    hidden_size: int
    layers: int
    heads: int
    head_dim: int
    rotary_dim: int
    q_lora_rank: int
    output_groups: int
    output_rank: int
    max_context: int
    epsilon: float
    streams: int
    hc_epsilon: float
    sinkhorn_iterations: int
    window: int
    rates: tuple[int, ...]
    kv_sources: tuple[int, ...]
    index_sources: tuple[int, ...]
    index_heads: int
    index_dim: int
    index_top_k: int
    candidate_source: int
    candidate_blocks: int
    candidate_block_size: int
    rope_main: Mapping[str, Any]
    rope_compress: Mapping[str, Any]
    experts: int
    top_k: int
    shared_experts: int
    moe_intermediate_size: int
    normalize_routes: bool
    routed_scale: float
    scoring: str
    swiglu_limit: float | None
    engram_layers: tuple[int, ...]
    engram_rows: tuple[int, ...]
    engram_ngram: int
    engram_heads: int
    engram_dim: int
    engram_vocab: int
    engram_compressed_vocab: int
    pad_id: int
    auxiliary_layers: int
    vision: bool
    source_quantization: Mapping[str, Any] | None
    family: str = "deepseek_v41"
    groups: int = 1
    topk_groups: int = 1

    @classmethod
    def from_hf(cls, source, *, max_context=None):
        family = source.get("model_type")
        if family not in ("deepseek_v41", "deepseek_v41_text"):
            _fail("V4.1 requires model_type deepseek_v41 or deepseek_v41_text")
        if source.get("architectures") not in (None, ["DeepseekV41ForCausalLM"]):
            _fail("architectures must identify DeepseekV41ForCausalLM")
        text = source.get("text_config", source)
        if not isinstance(text, Mapping) or text.get("model_type") not in ("deepseek_v41_text", "deepseek_v41"):
            _fail("V4.1 requires its own text_config")
        if source.get("vision_config") is not None and not isinstance(source["vision_config"], Mapping):
            _fail("vision_config must be an object when a text backbone is extracted")
        for name in ("attention_bias", "mlp_bias", "tie_word_embeddings", "is_encoder_decoder"):
            if _boolean(text, name):
                _fail(f"V4.1 {name} is unsupported")
        if text.get("hidden_act", "silu") != "silu" or text.get("attention_dropout", 0) != 0:
            _fail("V4.1 requires deterministic SiLU text inference")
        if text.get("topk_method", "noaux_tc") != "noaux_tc" or text.get("scoring_func", "sqrtsoftplus") != "sqrtsoftplus":
            _fail("V4.1 requires sqrtsoftplus/noaux_tc routing")
        if text.get("gate_temp", 1) != 1 or text.get("pretraining_tp", 1) != 1:
            _fail("V4.1 requires gate_temp=1 and pretraining_tp=1")
        integer = lambda key, default=None: positive_integer(text.get(key, default), key)
        h, layers, heads = integer("hidden_size"), integer("num_hidden_layers"), integer("num_attention_heads")
        d, r = integer("head_dim", 512), integer("qk_rope_head_dim", 64)
        if text.get("partial_rotary_factor") is not None and int(d * positive_float(text["partial_rotary_factor"], "partial_rotary_factor")) != r:
            _fail("V4.1 rotary width and partial_rotary_factor disagree")
        ih, di = integer("index_n_heads", 32), integer("index_head_dim", 128)
        groups = integer("o_groups", 8)
        if r % 2 or r > min(d, di) or heads % groups or integer("num_key_value_heads", 1) != 1:
            _fail("V4.1 rotary/group/shared-KV geometry is invalid")
        maximum = integer("max_position_embeddings")
        bound = maximum if max_context is None else positive_integer(max_context, "max_context")
        if bound > maximum:
            _fail("max_context exceeds the source context")
        auxiliary = _integer(text.get("num_nextn_predict_layers", 0), "num_nextn_predict_layers", zero=True)
        rates = text.get("compress_ratios")
        if (not isinstance(rates, (tuple, list)) or len(rates) not in (layers, layers + auxiliary) or
                any(type(v) is not int or v not in (0, 1, 2) for v in rates) or any(rates[layers:])):
            _fail("V4.1 requires one 0/1/2 compression ratio per layer and zero-compression draft stages")
        rates = tuple(rates[:layers])
        def layer_ids(key):
            value = text.get(key, [])
            if (not isinstance(value, (list, tuple)) or any(type(i) is not int or not 0 <= i < layers for i in value)
                    or list(value) != sorted(set(value))):
                _fail(f"{key} must contain sorted distinct backbone layer IDs")
            return tuple(value)
        kv, index = layer_ids("kv_source_layer_ids"), layer_ids("index_source_layer_ids")
        if not set(kv) <= set(index):
            _fail("every KV source must also produce index keys")
        current = None
        for i, rate in enumerate(rates):
            if i in kv:
                if not rate:
                    _fail("a KV source must have compressed attention")
                current = i
            if rate and (current is None or rate != rates[current]):
                _fail("compressed attention must share the current source's compression ratio")
            if i in index and (not rate or current is None):
                _fail("an index source requires a preceding compatible KV source")
        candidate = text.get("candidate_source_layer_id", -1)
        if type(candidate) is not int or candidate != -1 and candidate not in index:
            _fail("candidate_source_layer_id must identify an index source or -1")
        if candidate >= 0 and any(rates[i] != rates[candidate] for i in index if i > candidate):
            _fail("two-stage candidates require a stable compressed-position domain")
        rope_source = dict(text)
        scaling = dict(rope_source.get("rope_scaling") or {})
        if "original_max_position" in scaling:
            if "original_max_position_embeddings" in scaling:
                _fail("ambiguous original RoPE context")
            scaling["original_max_position_embeddings"] = scaling.pop("original_max_position")
        if scaling:
            rope_source["rope_scaling"] = scaling
        compress = _rope_parameters(rope_source, theta=text.get("compress_rope_theta", 160000), rotary=r)
        compress["attention_factor"] = 1.0
        main = {"rope_type": "default", "rope_theta": positive_float(text.get("rope_theta", 10000), "rope_theta"), "rotary_dim": r}
        experts, top = integer("n_routed_experts"), integer("num_experts_per_tok")
        if top > experts or integer("n_shared_experts", 1) != 1:
            _fail("V4.1 requires a valid routed bank and one shared expert")
        engram = layer_ids("engram_layer_ids")
        rows = text.get("engram_num_embeddings", [])
        if not isinstance(rows, (list, tuple)) or len(rows) != len(engram):
            _fail("Engram needs one table size per declared layer")
        rows = tuple(positive_integer(v, "engram_num_embeddings") for v in rows)
        if any(v > 2147483647 for v in rows):
            _fail("Engram row IDs exceed portable I32")
        ng = integer("engram_max_ngram_size", 4) if engram else 2
        if ng < 2:
            _fail("Engram n-grams must include at least two tokens")
        vocab = integer("vocab_size")
        compressed = integer("engram_compressed_vocab_size", vocab) if engram else vocab
        pad = _integer(text.get("engram_pad_token_id", 2), "engram_pad_token_id", zero=True)
        if compressed > vocab or compressed > 2147483647 or pad >= vocab:
            _fail("Engram compressed vocabulary or pad ID is invalid")
        quant = source.get("quantization_config", text.get("quantization_config"))
        if quant is not None and not isinstance(quant, Mapping):
            _fail("quantization_config must be an object")
        limit = text.get("swiglu_limit", 10)
        limit = None if limit is None or limit == 0 else positive_float(limit, "swiglu_limit")
        result = cls(vocab, h, layers, heads, d, r, integer("q_lora_rank"), groups, integer("o_lora_rank"),
            bound, positive_float(text.get("rms_norm_eps", 1e-20), "rms_norm_eps"), integer("hc_mult", 4),
            positive_float(text.get("hc_eps", 1e-6), "hc_eps"), integer("hc_sinkhorn_iters", 20), integer("sliding_window", 128),
            rates, kv, index, ih, di, integer("index_topk", 512), candidate, integer("candidate_topk_blocks", 2048),
            integer("candidate_block_size", 8), main, compress, experts, top, 1, integer("moe_intermediate_size"),
            _boolean(text, "norm_topk_prob", True) and top > 1, positive_float(text.get("routed_scaling_factor", 1.5), "routed_scaling_factor"),
            "sqrtsoftplus", limit, engram, rows, ng, integer("engram_n_heads", 8) if engram else 1, integer("engram_head_dim", 256) if engram else 1,
            integer("engram_vocab_size", 16000000) if engram else 1, compressed, pad, auxiliary,
            isinstance(source.get("vision_config"), Mapping), dict(quant) if quant else None)
        result.engram_tables()  # Validate the declared row domain before weight IO.
        return result

    def engram_tables(self):
        # Trial division is bounded by sqrt(I32); only a few dozen primes per
        # model are required. No training/library dependency is introduced.
        seen, tables = set(), {}
        def prime(value):
            return value >= 2 and (value == 2 or value % 2 and all(value % i for i in range(3, math.isqrt(value) + 1, 2)))
        for layer, rows in zip(self.engram_layers, self.engram_rows):
            sizes = []
            for _ in range(self.engram_ngram - 1):
                current = self.engram_vocab - 1
                for _ in range(self.engram_heads):
                    current += 1
                    while current in seen or not prime(current):
                        current += 1
                    seen.add(current); sizes.append(current)
            if sum(sizes) != rows:
                _fail(f"Engram layer {layer} declared rows disagree with its prime bucket layout")
            bound = max(1, (np.iinfo(np.int64).max // self.engram_compressed_vocab) // 2)
            multipliers = np.random.default_rng(10007 * layer).integers(0, bound,
                size=self.engram_ngram, dtype=np.int64) * 2 + 1
            tables[layer] = (multipliers, np.asarray(sizes, np.int32), np.cumsum([0, *sizes[:-1]], dtype=np.int64).astype(np.int32))
        return tables

    def weight_shapes(self):
        h, d, m = self.hidden_size, self.head_dim, self.streams
        result = {"embed.weight": (self.vocab_size, h), "head.weight": (self.vocab_size, h), "norm.weight": (h,)}
        for i in range(self.layers):
            p = f"layers.{i}."
            for role in ("attn", "ffn"):
                result[p + role + "_norm.weight"] = (h,)
                result[p + "hc_" + role + "_fn"] = (2*m + m*m, m*h)
                result[p + "hc_" + role + "_base"] = (2*m + m*m,)
                result[p + "hc_" + role + "_scale"] = (3,)
            a = p + "attn."
            result.update({a+"wq_a.weight": (self.q_lora_rank,h), a+"q_norm.weight": (self.q_lora_rank,),
                a+"wq_b.weight": (self.heads*d,self.q_lora_rank), a+"wkv.weight": (d,h), a+"kv_norm.weight": (d,),
                a+"wo_a.weight": (self.output_groups*self.output_rank,self.heads*d//self.output_groups),
                a+"wo_b.weight": (h,self.output_groups*self.output_rank), a+"attn_sink": (self.heads,)})
            if i in self.kv_sources:
                result[a+"compressor.wkv.weight"] = (d,h)
                result[a+"compressor.norm.weight"] = (d,)
                if self.rates[i] == 2:
                    result[a+"compressor.wgate.weight"] = (d,h)
            if i in self.index_sources:
                result[a+"indexer.wq_b.weight"] = (self.index_heads*self.index_dim,self.q_lora_rank)
                result[a+"indexer.weights_proj.weight"] = (self.index_heads,h)
                if i in self.kv_sources:
                    result[a+"indexer.wk.weight"] = (self.index_dim,d)
                    result[a+"indexer.k_norm.weight"] = (self.index_dim,)
            f = p+"ffn."
            result[f+"gate.weight"] = (self.experts,h)
            result[f+"gate.bias"] = (self.experts,)
            for expert in [*(f"experts.{j}" for j in range(self.experts)), "shared_experts"]:
                for key, shape in (("w1",(self.moe_intermediate_size,h)),("w3",(self.moe_intermediate_size,h)),("w2",(h,self.moe_intermediate_size))):
                    result[f+expert+"."+key+".weight"] = shape
            if i in self.engram_layers:
                rows = self.engram_rows[self.engram_layers.index(i)]
                e = p+"engram."
                result[e+"embed.weight"] = (rows,self.engram_dim)
                result[e+"wkv.weight"] = (h*(m+1),(self.engram_ngram-1)*self.engram_heads*self.engram_dim)
                result[e+"q_weight"] = result[e+"k_weight"] = (m,h)
        return result


def _checked_weights(config, source):
    expected = config.weight_shapes()
    missing, extra = set(expected)-set(source), set(source)-set(expected)
    omitted = set()
    for name in extra:
        parts = name.split(".")
        if (len(parts)>2 and parts[0]=="mtp" and parts[1].isdigit() and int(parts[1])<config.auxiliary_layers):
            omitted.add(name)
        elif config.vision and (name.startswith(("vision.","aligner.")) or name in ("image_start","image_end","image_newline") or
                len(parts)==5 and parts[0]=="layers" and parts[1].isdigit() and int(parts[1])<config.layers and parts[2:]==["ffn","gate","bias_vl"]):
            omitted.add(name)
    if missing or extra-omitted:
        _fail(f"V4.1 checkpoint namespace mismatch; missing={sorted(missing)}, extra={sorted(extra-omitted)}", "VXHF_WEIGHTS")
    result = {}
    for name, shape in expected.items():
        value = np.asarray(source[name])
        if value.dtype not in (np.dtype("float32"),np.dtype("float16")) or value.shape != shape or not np.isfinite(value).all():
            _fail(f"{name} requires finite floating shape {shape}; received {value.dtype} {value.shape}", "VXHF_WEIGHTS")
        result[name] = np.ascontiguousarray(value,dtype=np.float32)
    return result


def compressed_token_map(tokenizer_path, config):
    """Reproduce the official tokenizer normalization once, during export."""
    from tokenizers import Tokenizer, Regex, normalizers
    backend = Tokenizer.from_file(str(tokenizer_path))
    if backend.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        _fail("tokenizer vocabulary does not match V4.1", "VXHF_SOURCE")
    sentinel = "\ue000"
    normalize = normalizers.Sequence([normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),
        normalizers.Lowercase(),normalizers.Replace(Regex(r"[ \t\r\n]+")," "),
        normalizers.Replace(Regex(r"^ $"),sentinel),normalizers.Strip(),normalizers.Replace(sentinel," ")])
    mapping, seen = [], {}
    for token in range(config.vocab_size):
        text = backend.decode([token],skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else normalize.normalize_str(text) or text
        if key not in seen:
            seen[key] = len(seen)
        mapping.append(seen[key])
    if len(seen) != config.engram_compressed_vocab:
        _fail("tokenizer normalization disagrees with engram_compressed_vocab_size", "VXHF_SOURCE")
    return np.asarray(mapping,np.int32)


def build_deepseek_v41_graph(config, source_weights, *, token_map=None, output_name="logits",
                             capture_kv=False, compact_kv=True, cached=False, generate=False):
    if cached:
        _fail("V4.1 shared sparse-index row caching is not qualified; use full-prefix generation")
    if not isinstance(generate,bool) or not isinstance(cached,bool) or not isinstance(output_name,str) or not is_runtime_graph_name(output_name):
        _fail("invalid V4.1 execution/output options")
    weights = _checked_weights(config,source_weights)
    outputs = [output_name] + ([f"layer.{i}.{part}" for i in range(config.layers) for part in ("k_rope","v")] if capture_kv else [])
    b = GraphBuilder(config.max_context,outputs)
    for name,value in weights.items(): b.add(name,value)
    s,h,m,d,r = "S",config.hidden_size,config.streams,config.head_dim,config.rotary_dim
    hs, streams = [1,s,h],[1,s,m,h]
    def expand(name,value,shape,dtype="float32"):
        return b.tensor(name,"Expand",{"input":value},shape,{"shape":shape},dtype)
    def gather(name,value,indices,axis,shape,dtype="float32"):
        index = b.add(name+".indices",np.asarray(indices,np.int32))
        return b.tensor(name,"Gather",{"input":value,"indices":index},shape,{"axis":axis},dtype)
    def linear(name,value,weight,width): return b.linear(name,value,weight,width)
    pairs = {}
    def pair(rate):
        if rate not in pairs:
            parameters = config.rope_compress if rate else config.rope_main
            tables = _rotary_tables(config,parameters,interleaved=True,rate=rate or 1)
            pair_name = "hf.v41.rope."+str(rate)
            pairs[rate] = tuple(b.add(pair_name+"."+part,value) for part,value in zip(("cos","sin"),tables))
        return pairs[rate]
    query_pairs = {}
    def query_pair(compressed):
        if compressed not in query_pairs:
            # Queries use their absolute token position, never compressed rows.
            tables = pair(1 if compressed else 0)
            prefix = "hf.v41.query."+str(int(compressed))
            query_pairs[compressed] = tuple(b.reshape(prefix+"."+part+".rows",
                b.tensor(prefix+"."+part,"Embedding",{"input":"position_ids","weight":table},[s,r//2]),[1,s,r//2])
                for part,table in zip(("cos","sin"),tables))
        return query_pairs[compressed]
    def rotary(name,value,heads,width,tables):
        return b.tensor(name,"RotaryEmbedding",{"input":value,"cos":tables[0],"sin":tables[1]},[1,s,heads,width],
            {"rotary_dim":r,"position_offset":width-r,"interleaved":True})
    def compressed_pair(rate):
        # Padded compressed rows are laid out from zero; query position IDs do
        # not index these tables. Only the completed prefix is attended to.
        tables = pair(rate)
        return tuple(b.reshape("hf.v41.compressed."+str(rate)+"."+part+".rows",
            b.tensor("hf.v41.compressed."+str(rate)+"."+part,"Embedding",{"input":"position_ids","weight":table},[s,r//2]),[1,s,r//2])
            for part,table in zip(("cos","sin"),tables))
    compressed_pairs = {}
    # Whole-prefix positions are zero-based/contiguous (the execution contract
    # generates these); compression and the indexer use that same causal domain.
    def get_compressed_pair(rate):
        if rate not in compressed_pairs: compressed_pairs[rate] = compressed_pair(rate)
        return compressed_pairs[rate]
    tables = config.engram_tables()
    if config.engram_layers:
        token_map = np.asarray(token_map)
        if token_map.dtype.kind not in "iu" or token_map.shape != (config.vocab_size,) or np.any(token_map<0) or np.any(token_map>=config.engram_compressed_vocab) or len(np.unique(token_map)) != config.engram_compressed_vocab:
            _fail("Engram requires an exact complete compressed token map", "VXHF_WEIGHTS")
        b.add("hf.v41.token_map",token_map.astype(np.int32))
        compressed_tokens = b.tensor("hf.v41.compressed_tokens","Gather",{"input":"hf.v41.token_map","indices":"token_ids"},[1,s],{"axis":0},dtype="int32")
    def engram(layer,hidden):
        p,name = f"layers.{layer}.engram.",f"layer.{layer}.engram"
        multipliers,sizes,offsets = tables[layer]
        words=np.stack((multipliers.astype(np.uint64)&np.uint64(0xffffffff),multipliers.astype(np.uint64)>>np.uint64(32)),axis=1).astype(np.uint32).view(np.int32)
        for suffix,value in (("multipliers",words),("sizes",sizes),("offsets",offsets)): b.add(p+suffix,value)
        cols=(config.engram_ngram-1)*config.engram_heads
        ids=b.tensor(name+".ids","NGramIds",{"token_ids":compressed_tokens,"multipliers":p+"multipliers","vocab_sizes":p+"sizes","offsets":p+"offsets"},
            [1,s,cols],{"ngram_size":config.engram_ngram,"heads_per_ngram":config.engram_heads,"eos_token_id":-1,
                "pad_token_id":int(token_map[config.pad_id]),"vocab_size":config.engram_compressed_vocab},"int32")
        embedded=b.tensor(name+".lookup","Embedding",{"input":ids,"weight":p+"embed.weight"},[1,s,cols,config.engram_dim])
        embedded=b.reshape(name+".features",embedded,[1,s,cols*config.engram_dim])
        projected=linear(name+".kv",embedded,p+"wkv.weight",h*(m+1))
        key=gather(name+".key",projected,range(m*h),2,[1,s,m*h])
        key=b.reshape(name+".key.streams",key,streams)
        unit=b.add(name+".unit",np.ones(h,np.float32))
        query=b.norm(name+".query.norm",hidden,unit,streams,config.epsilon)
        key=b.norm(name+".key.norm",key,unit,streams,config.epsilon)
        query=b.tensor(name+".query.weighted","Mul",{"a":query,"b":expand(name+".q_weight",p+"q_weight",streams)},streams)
        key=b.tensor(name+".key.weighted","Mul",{"a":key,"b":expand(name+".k_weight",p+"k_weight",streams)},streams)
        product=b.tensor(name+".product","Mul",{"a":query,"b":key},streams)
        gate=b.tensor(name+".dot","ReduceSum",{"input":product},[1,s,m,1],{"axis":-1,"keepdims":True})
        divisor=expand(name+".divisor.expand",b.add(name+".divisor",np.array([math.sqrt(h)],np.float32)),[1,s,m,1])
        gate=b.tensor(name+".scaled","Div",{"a":gate,"b":divisor},[1,s,m,1])
        gate=b.tensor(name+".root","SignedSqrt",{"input":gate},[1,s,m,1],{"eps":1e-6})
        gate=b.tensor(name+".gate","Sigmoid",{"input":gate},[1,s,m,1])
        value=gather(name+".value",projected,range(m*h,(m+1)*h),2,hs)
        value=expand(name+".value.expand",b.reshape(name+".value.rows",value,[1,s,1,h]),streams)
        gate=expand(name+".gate.expand",gate,streams)
        update=b.tensor(name+".update","Mul",{"a":value,"b":gate},streams)
        return b.tensor(name,"Add",{"a":hidden,"b":update},streams)
    shared_kv=shared_index_k=shared_mask=shared_candidates=None
    def attention(layer,x):
        nonlocal shared_kv,shared_index_k,shared_mask,shared_candidates
        p,name=f"layers.{layer}.attn.",f"layer.{layer}.attn"
        rate=config.rates[layer]; qp=query_pair(bool(rate))
        qa=linear(name+".qa",x,p+"wq_a.weight",config.q_lora_rank)
        qa=b.norm(name+".qa.norm",qa,p+"q_norm.weight",[1,s,config.q_lora_rank],config.epsilon)
        q=linear(name+".qb",qa,p+"wq_b.weight",config.heads*d)
        q=rotary(name+".q.rope",b.reshape(name+".q.heads",q,[1,s,config.heads,d]),config.heads,d,qp)
        kv=linear(name+".kv",x,p+"wkv.weight",d)
        kv=b.norm(name+".kv.norm",kv,p+"kv_norm.weight",[1,s,d],config.epsilon)
        kv=rotary(f"layer.{layer}.k_rope",b.reshape(name+".kv.heads",kv,[1,s,1,d]),1,d,qp)
        b.reshape(f"layer.{layer}.v",kv,[1,s,1,d])
        if layer in config.kv_sources:
            c=p+"compressor."
            latent=linear(name+".compress.kv",x,c+"wkv.weight",d)
            gate=linear(name+".compress.gate",x,c+"wgate.weight",d) if rate==2 else expand(name+".compress.gate",b.add(name+".zero",np.array([0],np.float32)),[1,s,d])
            bias=b.add(name+".compress.bias",np.zeros((rate,d),np.float32))
            latent=b.tensor(name+".compressed.latent","KVCompress",{"input":latent,"a":gate,"bias":bias,"weight":c+"norm.weight"},[1,s,d],
                {"num_groups":1,"compress_rate":rate,"eps":config.epsilon})
            ik=linear(name+".index.k",latent,p+"indexer.wk.weight",config.index_dim)
            ik=b.norm(name+".index.k.norm",ik,p+"indexer.k_norm.weight",[1,s,config.index_dim],config.epsilon)
            ik=rotary(name+".index.k.rope",b.reshape(name+".index.k.head",ik,[1,s,1,config.index_dim]),1,config.index_dim,get_compressed_pair(rate))
            shared_index_k=b.reshape(name+".index.k.rows",ik,[1,s,config.index_dim])
            ck=rotary(name+".compressed.rope",b.reshape(name+".compressed.head",latent,[1,s,1,d]),1,d,get_compressed_pair(rate))
            shared_kv=b.reshape(name+".compressed.rows",ck,[1,s,d])
        if layer in config.index_sources:
            iq=linear(name+".index.q",qa,p+"indexer.wq_b.weight",config.index_heads*config.index_dim)
            iq=rotary(name+".index.q.rope",b.reshape(name+".index.q.heads",iq,[1,s,config.index_heads,config.index_dim]),config.index_heads,config.index_dim,qp)
            iw=linear(name+".index.weights",x,p+"indexer.weights_proj.weight",config.index_heads)
            ins={"q":iq,"k":shared_index_k,"index_weights":iw}
            params={"compress_rate":rate,"top_k":config.index_top_k}
            outs={"out":(name+".index.mask",[1,s,s],"int32")}
            if layer==config.candidate_source:
                params.update(candidate_block_size=config.candidate_block_size,candidate_top_k=config.candidate_blocks)
                outs["candidates"]=(name+".index.candidates",[1,s,s],"int32")
            elif config.candidate_source>=0 and layer>config.candidate_source: ins["candidates"]=shared_candidates
            selected=b.multi(name+".index","SparseIndexer",ins,outs,params)
            shared_mask=selected["out"]
            if "candidates" in selected: shared_candidates=selected["candidates"]
        ins={"q":q,"k":kv,"v":kv,"mask":"attention_mask","position_ids":"position_ids","sinks":p+"attn_sink"}
        params={"causal":True,"scale":d**-0.5,"window_size":config.window}
        if rate:
            ins.update(compressed_kv=shared_kv,compressed_mask=shared_mask);params["compress_rate"]=rate
        out=b.tensor(name+".attention","SparseAttention",ins,[1,s,config.heads,d],params)
        minus=expand(name+".minus.expand",b.add(name+".minus",np.array([-1],np.float32)),[1,s,r//2])
        negsin=b.tensor(name+".inverse.sin","Mul",{"a":qp[1],"b":minus},[1,s,r//2])
        out=rotary(name+".inverse",out,config.heads,d,(qp[0],negsin))
        flat=b.reshape(name+".flat",out,[1,s,config.heads*d])
        width=config.heads*d//config.output_groups; parts=[]
        for group in range(config.output_groups):
            part=gather(name+f".group.{group}",flat,range(group*width,(group+1)*width),2,[1,s,width])
            w=b.add(name+f".group.{group}.weight",weights[p+"wo_a.weight"][group*config.output_rank:(group+1)*config.output_rank])
            parts.append(linear(name+f".group.{group}.out",part,w,config.output_rank))
        out=parts[0]
        for group in range(1,len(parts)):
            out=b.tensor(name+f".join.{group}","Concat2",{"a":out,"b":parts[group]},[1,s,(group+1)*config.output_rank],{"axis":2})
        return linear(name+".out",out,p+"wo_b.weight",h)
    hidden=b.tensor("embedding","Embedding",{"input":"token_ids","weight":"embed.weight"},hs)
    hidden=expand("embedding.streams",b.reshape("embedding.rows",hidden,[1,s,1,h]),streams)
    identity=np.zeros((m,),np.float32);identity[0]=1
    pre=expand("hf.v41.identity.pre",b.add("hf.v41.identity",identity),[1,s,m])
    for layer in range(config.layers):
        if layer in config.engram_layers: hidden=engram(layer,hidden)
        p,name=f"layers.{layer}.",f"layer.{layer}"
        for role in ("attn","ffn"):
            connection=b.multi(name+"."+role+".hc","HyperConnection",{"input":hidden,"weight":p+"hc_"+role+"_fn",
                "bias":p+"hc_"+role+"_base","scale":p+"hc_"+role+"_scale"},
                {"pre":(name+"."+role+".pre",[1,s,m],"float32"),"post":(name+"."+role+".post",[1,s,m],"float32"),
                 "comb":(name+"."+role+".comb",[1,s,m,m],"float32")},
                {"eps":config.epsilon,"normalization_epsilon":config.hc_epsilon,"iterations":config.sinkhorn_iterations})
            x=collapse_streams(b,name+"."+role+".input",hidden,pre,m,h)
            x=b.norm(name+"."+role+".norm",x,p+role+"_norm.weight",hs,config.epsilon)
            if role=="attn": update=attention(layer,x)
            else:
                # Reuse one MoE lowering, adapting only the official names.
                fp=p+"ffn."; common={}
                for key,value in weights.items():
                    if key.startswith(fp):
                        target=key.replace(".w1.",".gate_proj.").replace(".w3.",".up_proj.").replace(".w2.",".down_proj.")
                        if target.endswith(".gate.bias"): target=target[:-4]+"e_score_correction_bias"
                        common[target]=value
                        if target not in b.weights:b.add(target,value)
                update=add_deepseek_moe(b,config,common,name+".mlp",x,fp,layer)
            hidden=b.tensor(name+"."+role+".residual","HyperResidual",{"input":hidden,"update":update,
                "post":connection["post"],"comb":connection["comb"]},streams)
            pre=connection["pre"]
    hidden=collapse_streams(b,"model.hyper.head",hidden,pre,m,h)
    hidden=b.norm("model.final.norm",hidden,"norm.weight",hs,config.epsilon)
    linear(output_name,hidden,"head.weight",config.vocab_size)
    if generate:
        from .execution_contract import full_prefix_decoder_execution
        if output_name=="generated":_fail("logits output conflicts with generated IDs")
        b.tensor("generated","ArgMax",{"input":output_name},[1,s],{"axis":2,"keepdims":False,"select_last_index":0},"int32")
        b.graph["outputs"].append("generated")
        return b.finish(full_prefix_decoder_execution(b.graph["inputs"],b.graph["outputs"],token_input="token_ids",
            token_output="generated",keep_input="attention_mask",position_input="position_ids",logits_output=output_name,id_domain_size=config.vocab_size))
    return b.finish()


def export_hf_deepseek_v41(directory,out_path,*,max_context=None,output_names=None,capture_kv=False,
                           compact_kv=False,cached=False,generate=False,report_callback=None,weight_precision=None, group_size=128):
    if max_context is None:_fail("V4.1 export requires explicit bounded max_context")
    if cached:_fail("V4.1 shared sparse-index row caching is not qualified; use full-prefix generation")
    if not isinstance(cached,bool) or not isinstance(generate,bool):_fail("cached and generate must be boolean")
    if output_names is not None and len(output_names)!=1:_fail("V4.1 export has exactly one logits output")
    source=read_config(Path(directory)/"config.json")
    config=DeepSeekV41Config.from_hf(source,max_context=max_context)
    token_map=compressed_token_map(Path(directory)/"tokenizer.json",config) if config.engram_layers else None
    _,weights,storage=read_checkpoint_storage(directory,config_validator=DeepSeekV41Config.from_hf)
    graph,initializers=build_deepseek_v41_graph(config,weights,token_map=token_map,
        output_name=output_names[0] if output_names else "logits",capture_kv=capture_kv,compact_kv=compact_kv,cached=cached,generate=generate)
    graph=write_hf_package(out_path,graph,initializers,weight_precision=weight_precision, group_size=group_size,
        storage_conversion="F32/F16/BF16;FP8-E4M3FN/FP4-E2M1+UE8M0->F32")
    if report_callback:
        report_callback({"features":{"hf_decoder":["family:deepseek_v41","execution:F32","batch:1",
            "attention:shared-csa2-two-stage-index","hyper-connections:delayed-pre-mix","engram:text-token-normalization",
            "state:complete-stateless-prefix","generation:full-prefix" if generate else "generation:undeclared",
            f"context-bound:{config.max_context}","performance:unqualified"],"source_storage":sorted(set(storage.values())),
            "source_quantization_config":config.source_quantization,"activation_fake_quantization":"not-emulated;F32-reference-scope",
            "vision":"excluded;text-only","auxiliary_MTP_layers":"excluded;ordinary-autoregressive-decoder"}})
    return graph
