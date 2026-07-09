//! VolvoxAI Synurang runtime — a single cdylib (`libvolvoxai.so`) that wraps the
//! C inference engine. Rust here is only glue: it implements the generated
//! `VolvoxAIServicePlugin` trait and calls the engine over `extern "C"`. The
//! whole engine (native/*.c) is cc-compiled into this same library by build.rs.
//!
//! The engine is a global singleton (native/engine.c), so every call is
//! serialized behind one mutex and only one model is loaded at a time; `model_id`
//! is a validation token over that single context.

use std::ffi::{c_char, c_int, c_long, CStr, CString};
use std::os::raw::c_void;
use std::ptr::addr_of;
use std::sync::Mutex;

// prost-generated protobuf messages (package volvoxai.v1).
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/volvoxai.v1.rs"));
    // prost lower-camels the CTC acronym; the Synurang generator keeps CTCRequest.
    pub type CTCRequest = CtcRequest;
    pub type CTCResponse = CtcResponse;
    // The RPCs use google.protobuf.Empty, which prost maps away; the generated
    // code refers to `Empty` as a concrete type, so define one here.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Empty {}
}

// Synurang-generated plugin trait + dispatcher + Synurang_* FFI exports. Loaded
// as a module file (not include!) so its leading inner attributes are legal;
// `make proto_rust` injects `use super::pb::*;` + `use std::boxed::Box;` at the
// top so its unqualified type names resolve here.
#[path = "gen/volvoxai_ffi_plugin.rs"]
mod ffi;

use ffi::{
    register_volvox_a_i_service_plugin, FfiError, PluginStreamSender, VolvoxAIServicePlugin,
};
use pb::*;

// ---------------------------------------------------------------------------
// C engine ABI (native/engine.h, native/tokenizer.h) + backend-select globals.
// ---------------------------------------------------------------------------
extern "C" {
    fn engine_init(config: *const c_char, weights: *const c_char) -> c_int;
    fn engine_free_ctx();
    fn engine_forward() -> c_int;
    fn engine_prefill(n: c_int) -> c_int;
    fn engine_decode(pos: c_int) -> c_int;
    fn engine_input_ptr(name: *const c_char, numel: *mut c_long) -> *mut f32;
    fn engine_tensor_ptr(name: *const c_char, numel: *mut c_long) -> *mut f32;
    fn engine_tensor_info(
        name: *const c_char,
        numel: *mut c_long,
        shape: *mut c_int,
        ndim: *mut c_int,
    ) -> c_int;
    fn engine_last_logits(count: *mut c_int) -> *const f32;

    fn tokenizer_init(vocab: *const c_char, merges: *const c_char) -> *mut c_void;
    fn tokenizer_free(t: *mut c_void);
    fn tokenizer_encode(
        t: *mut c_void,
        text: *const c_char,
        tokens: *mut c_int,
        max: c_int,
    ) -> c_int;
    fn tokenizer_decode(t: *mut c_void, id: c_int) -> *const c_char;

    static mut g_nn: c_int;
    static mut g_first_input: [c_char; 128];
}

// These globals are declared `extern` by the C engine and were historically
// defined by native/main.c. The runtime intentionally does not compile main.c,
// so the shared library must provide the definitions itself.
#[no_mangle]
pub static mut g_use_vulkan: c_int = 0;
#[no_mangle]
pub static mut g_use_opengl: c_int = 0;
#[no_mangle]
pub static mut g_use_metal: c_int = 0;
#[no_mangle]
pub static mut g_use_nnapi: c_int = 0;
#[no_mangle]
pub static mut g_debug: c_int = 0;
#[no_mangle]
pub static mut g_last_token: c_int = -1;

const MAX_SEQ: usize = 4096;
const DTYPE_F32: i32 = 1; // volvoxai.v1.DType.DTYPE_F32

// gRPC status codes carried in FfiError.grpc_code.
const INVALID_ARGUMENT: i32 = 3;
const NOT_FOUND: i32 = 5;
const UNIMPLEMENTED: i32 = 12;
const INTERNAL: i32 = 13;

fn err(msg: impl Into<String>, code: i32) -> FfiError {
    FfiError::new(msg, code, code)
}

// Storage size of a volvoxai.v1.DType value (0 = sub-byte / unhandled).
fn dtype_bytes(dt: i32) -> usize {
    match dt {
        1 | 10 | 18 => 4,    // F32, I32, U32
        2 | 3 | 9 | 17 => 2, // F16, BF16, I16, U16
        8 | 16 | 24 => 1,    // I8, U8, BOOL
        4 | 11 | 19 => 8,    // F64, I64, U64
        _ => 0,
    }
}

fn dtype_name(dt: i32) -> &'static str {
    match dt {
        0 => "UNSPECIFIED",
        1 => "F32",
        2 => "F16",
        3 => "BF16",
        4 => "F64",
        8 => "I8",
        9 => "I16",
        10 => "I32",
        11 => "I64",
        12 => "I4",
        16 => "U8",
        17 => "U16",
        18 => "U32",
        19 => "U64",
        20 => "U4",
        24 => "BOOL",
        _ => "UNKNOWN",
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac = frac;
            let mut exp = -14i32;
            while (frac & 0x0400) == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x03ff;
            sign | (((exp + 127) as u32) << 23) | (frac << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

fn read_le<const N: usize>(data: &[u8], i: usize) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&data[i * N..(i + 1) * N]);
    out
}

unsafe fn copy_tensor_to_input(t: &Tensor) -> Result<(), FfiError> {
    let cname =
        CString::new(t.name.as_str()).map_err(|_| err("bad input name", INVALID_ARGUMENT))?;
    let mut numel: c_long = 0;
    let dst = engine_input_ptr(cname.as_ptr(), &mut numel);
    if dst.is_null() {
        return Err(err(
            format!("unknown input tensor: {}", t.name),
            INVALID_ARGUMENT,
        ));
    }
    if numel < 0 {
        return Err(err(format!("input {} has invalid size", t.name), INTERNAL));
    }
    let numel = numel as usize;
    let width = dtype_bytes(t.dtype);
    if width == 0 {
        return Err(err(
            format!(
                "input {} has unsupported dtype {}",
                t.name,
                dtype_name(t.dtype)
            ),
            INVALID_ARGUMENT,
        ));
    }
    let want = numel
        .checked_mul(width)
        .ok_or_else(|| err(format!("input {} is too large", t.name), INVALID_ARGUMENT))?;
    if t.data.len() != want {
        return Err(err(
            format!(
                "input {} byte-size mismatch: got {}, want {} for {} elements of {}",
                t.name,
                t.data.len(),
                want,
                numel,
                dtype_name(t.dtype)
            ),
            INVALID_ARGUMENT,
        ));
    }

    let dst = std::slice::from_raw_parts_mut(dst, numel);
    match t.dtype {
        1 => std::ptr::copy_nonoverlapping(t.data.as_ptr(), dst.as_mut_ptr() as *mut u8, want),
        2 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = f16_to_f32(u16::from_le_bytes(read_le::<2>(&t.data, i)));
            }
        }
        3 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = f32::from_bits((u16::from_le_bytes(read_le::<2>(&t.data, i)) as u32) << 16);
            }
        }
        4 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = f64::from_le_bytes(read_le::<8>(&t.data, i)) as f32;
            }
        }
        8 => {
            for (out, &x) in dst.iter_mut().zip(&t.data) {
                *out = (x as i8) as f32;
            }
        }
        9 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = i16::from_le_bytes(read_le::<2>(&t.data, i)) as f32;
            }
        }
        10 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = i32::from_le_bytes(read_le::<4>(&t.data, i)) as f32;
            }
        }
        11 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = i64::from_le_bytes(read_le::<8>(&t.data, i)) as f32;
            }
        }
        16 | 24 => {
            for (out, &x) in dst.iter_mut().zip(&t.data) {
                *out = x as f32;
            }
        }
        17 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = u16::from_le_bytes(read_le::<2>(&t.data, i)) as f32;
            }
        }
        18 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = u32::from_le_bytes(read_le::<4>(&t.data, i)) as f32;
            }
        }
        19 => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = u64::from_le_bytes(read_le::<8>(&t.data, i)) as f32;
            }
        }
        _ => {
            return Err(err(
                format!(
                    "input {} has unsupported dtype {}",
                    t.name,
                    dtype_name(t.dtype)
                ),
                INVALID_ARGUMENT,
            ))
        }
    }
    Ok(())
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as i32
}

// Apply ExecOptions onto the engine's backend-select globals (partial update).
unsafe fn apply_exec(exec: &Option<ExecOptions>) {
    if let Some(e) = exec {
        if let Some(b) = e.backend {
            g_use_vulkan = 0;
            g_use_opengl = 0;
            g_use_metal = 0;
            g_use_nnapi = 0;
            match b {
                2 => g_use_vulkan = 1,
                3 => g_use_opengl = 1,
                4 => g_use_metal = 1,
                5 => g_use_nnapi = 1,
                _ => {}
            }
        }
        if let Some(d) = e.debug {
            g_debug = if d { 1 } else { 0 };
        }
    }
}

// Copy a materialized F32 graph tensor into a proto Tensor.
unsafe fn read_tensor(name: &str) -> Result<Tensor, FfiError> {
    let cname = CString::new(name).map_err(|_| err("bad tensor name", INVALID_ARGUMENT))?;
    let mut numel: c_long = 0;
    let mut shape = [0 as c_int; 8];
    let mut ndim: c_int = 0;
    if engine_tensor_info(cname.as_ptr(), &mut numel, shape.as_mut_ptr(), &mut ndim) != 0 {
        return Err(err(format!("no such tensor: {name}"), NOT_FOUND));
    }
    let p = engine_tensor_ptr(cname.as_ptr(), &mut numel);
    if p.is_null() {
        return Err(err(format!("tensor has no data: {name}"), INTERNAL));
    }
    let data = std::slice::from_raw_parts(p as *const u8, numel as usize * 4).to_vec();
    Ok(Tensor {
        name: name.to_string(),
        shape: shape[..ndim as usize].iter().map(|&d| d as i64).collect(),
        dtype: DTYPE_F32,
        data,
        quant: None,
    })
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------
struct Inner {
    model_id: Option<String>,
    info: Option<ModelInfo>,
    tok: *mut c_void,
    counter: u64,
}
// tok is a raw pointer, only ever touched under the mutex.
unsafe impl Send for Inner {}

struct Plugin {
    inner: Mutex<Inner>,
}

impl Plugin {
    fn new() -> Self {
        Plugin {
            inner: Mutex::new(Inner {
                model_id: None,
                info: None,
                tok: std::ptr::null_mut(),
                counter: 0,
            }),
        }
    }
}

fn require(inner: &Inner, id: &str) -> Result<(), FfiError> {
    match &inner.model_id {
        Some(m) if m == id => Ok(()),
        _ => Err(err(
            format!("unknown or unloaded model_id: {id}"),
            NOT_FOUND,
        )),
    }
}

impl VolvoxAIServicePlugin for Plugin {
    fn ping(&self, _request: Empty) -> Result<PingResponse, FfiError> {
        Ok(PingResponse {
            version: "volvoxai-runtime/0.1".into(),
            available_backends: vec![1], // BACKEND_CPU
        })
    }

    fn get_capabilities(&self, _request: Empty) -> Result<Capabilities, FfiError> {
        // TODO: emit the docs/operation_list.md matrix per backend/op.
        Ok(Capabilities {
            version: "volvoxai-runtime/0.1".into(),
            backends: vec![BackendCapability {
                backend: 1, // CPU
                available: true,
                ops: vec![],
            }],
        })
    }

    fn load_model(&self, request: LoadModelRequest) -> Result<LoadModelResponse, FfiError> {
        let mut inner = self.inner.lock().unwrap();
        let paths = match request.source {
            Some(load_model_request::Source::Paths(p)) => p,
            Some(load_model_request::Source::Inline(_)) => {
                return Err(err(
                    "inline model bytes not supported yet; use paths",
                    UNIMPLEMENTED,
                ))
            }
            None => return Err(err("missing model source", INVALID_ARGUMENT)),
        };
        unsafe {
            apply_exec(&request.exec);
            if !inner.tok.is_null() {
                tokenizer_free(inner.tok);
                inner.tok = std::ptr::null_mut();
            }
            let cfg = CString::new(paths.config_path.as_str())
                .map_err(|_| err("bad config path", INVALID_ARGUMENT))?;
            let w = CString::new(paths.weights_path.as_str())
                .map_err(|_| err("bad weights path", INVALID_ARGUMENT))?;
            if engine_init(cfg.as_ptr(), w.as_ptr()) != 0 {
                return Err(err("engine_init failed", INTERNAL));
            }
            if !paths.tokenizer_path.is_empty() {
                if let Ok(tp) = CString::new(paths.tokenizer_path.as_str()) {
                    inner.tok = tokenizer_init(tp.as_ptr(), std::ptr::null());
                }
            }
            inner.counter += 1;
            let id = format!("model-{}", inner.counter);
            let info = ModelInfo {
                inputs: vec![],
                outputs: vec![],
                num_ops: g_nn,
                supports_kv_cache: !inner.tok.is_null(),
                has_tokenizer: !inner.tok.is_null(),
                op_counts: vec![],
            };
            inner.model_id = Some(id.clone());
            inner.info = Some(info.clone());
            Ok(LoadModelResponse {
                model_id: id,
                info: Some(info),
            })
        }
    }

    fn unload_model(&self, request: ModelRef) -> Result<Empty, FfiError> {
        let mut inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        unsafe {
            if !inner.tok.is_null() {
                tokenizer_free(inner.tok);
                inner.tok = std::ptr::null_mut();
            }
            engine_free_ctx();
        }
        inner.model_id = None;
        inner.info = None;
        Ok(Empty {})
    }

    fn get_model_info(&self, request: ModelRef) -> Result<ModelInfo, FfiError> {
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        Ok(inner.info.clone().unwrap_or_default())
    }

    fn list_models(&self, _request: Empty) -> Result<ListModelsResponse, FfiError> {
        let inner = self.inner.lock().unwrap();
        let mut r = ListModelsResponse::default();
        if let Some(id) = &inner.model_id {
            r.model_ids.push(id.clone());
            if let Some(info) = &inner.info {
                r.models.push(info.clone());
            }
        }
        Ok(r)
    }

    fn run(&self, request: RunRequest) -> Result<RunResponse, FfiError> {
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        unsafe {
            apply_exec(&request.exec);
            for t in &request.inputs {
                copy_tensor_to_input(t)?;
            }
            g_last_token = request.last_token.unwrap_or(-1);
            if engine_forward() != 0 {
                return Err(err("engine_forward failed", INTERNAL));
            }
            let mut outputs = Vec::new();
            if request.output_names.is_empty() {
                let mut count: c_int = 0;
                let lg = engine_last_logits(&mut count);
                if !lg.is_null() {
                    let data =
                        std::slice::from_raw_parts(lg as *const u8, count as usize * 4).to_vec();
                    outputs.push(Tensor {
                        name: "logits".into(),
                        shape: vec![count as i64],
                        dtype: DTYPE_F32,
                        data,
                        quant: None,
                    });
                }
            } else {
                for n in &request.output_names {
                    outputs.push(read_tensor(n)?);
                }
            }
            Ok(RunResponse {
                outputs,
                timing: None,
            })
        }
    }

    // Streaming greedy (argmax) decode; mirrors native/main.c generate. Sampling
    // (temperature/top_p/top_k) and VLM image input are TODO.
    fn generate(
        &self,
        request: GenerateRequest,
        stream: &dyn PluginStreamSender<GenerateEvent>,
    ) -> Result<(), FfiError> {
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        if inner.tok.is_null() {
            return Err(err(
                "model has no tokenizer; Generate needs one",
                UNIMPLEMENTED,
            ));
        }
        if request.image.is_some() {
            return Err(err("VLM/chat image input is a TODO", UNIMPLEMENTED));
        }
        unsafe {
            apply_exec(&request.exec);
            let cfg = request.r#gen.unwrap_or_default();
            let max_new = cfg.max_new_tokens.unwrap_or(50);
            let eos = cfg.eos_token.unwrap_or(-1);
            let pad = cfg.pad_token.unwrap_or(50256);

            let prompt = CString::new(request.prompt.as_str())
                .map_err(|_| err("bad prompt", INVALID_ARGUMENT))?;
            let token_name = CString::new("tokens").unwrap();
            let mut token_numel: c_long = 0;
            let mut token_input = engine_input_ptr(token_name.as_ptr(), &mut token_numel);
            if token_input.is_null() {
                token_input = engine_input_ptr(addr_of!(g_first_input) as *const c_char, &mut token_numel);
            }
            if token_input.is_null() || token_numel <= 0 {
                return Err(err("model has no token input", INVALID_ARGUMENT));
            }
            let cap = (token_numel as usize).min(MAX_SEQ);
            let mut ids = vec![0 as c_int; MAX_SEQ];
            let n = tokenizer_encode(
                inner.tok,
                prompt.as_ptr(),
                ids.as_mut_ptr(),
                cap as c_int,
            );
            if n <= 0 {
                return Err(err("prompt encoded to zero tokens", INVALID_ARGUMENT));
            }
            for i in 0..cap {
                *token_input.add(i) = if i < n as usize {
                    ids[i] as f32
                } else {
                    pad as f32
                };
            }
            if engine_prefill(n) != 0 {
                return Err(err("engine_prefill failed", INTERNAL));
            }

            let mut full = String::new();
            let mut reason = 2; // FINISH_LENGTH
            let mut pos = n - 1;
            let mut generated = 0;
            for _ in 0..max_new {
                if pos as usize >= cap - 1 {
                    break;
                }
                if stream.is_cancelled() {
                    reason = 3; // FINISH_STOP
                    break;
                }
                let mut count: c_int = 0;
                let lg = engine_last_logits(&mut count);
                if lg.is_null() {
                    return Err(err("no logits after decode", INTERNAL));
                }
                let logits = std::slice::from_raw_parts(lg, count as usize);
                let next = argmax(logits);
                let piece = tokenizer_decode(inner.tok, next);
                let text = if piece.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(piece).to_string_lossy().into_owned()
                };
                full.push_str(&text);

                let ev = GenerateEvent {
                    event: Some(generate_event::Event::Token(Token {
                        id: next,
                        text,
                        position: pos + 1,
                    })),
                };
                generated += 1;
                if !stream.send(ev) {
                    reason = 3;
                    break;
                }
                if next == eos {
                    reason = 1; // FINISH_EOS
                    break;
                }
                // Feed the sampled token and advance one position.
                pos += 1;
                *token_input.add(pos as usize) = next as f32;
                if engine_decode(pos) != 0 {
                    return Err(err("engine_decode failed", INTERNAL));
                }
            }

            let done = GenerateDone {
                text: full,
                num_tokens: generated,
                reason,
                timing: None,
            };
            stream.send(GenerateEvent {
                event: Some(generate_event::Event::Done(done)),
            });
            Ok(())
        }
    }

    fn get_tensor(&self, request: GetTensorRequest) -> Result<Tensor, FfiError> {
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        unsafe { read_tensor(&request.name) }
    }

    fn classify(&self, request: ClassifyRequest) -> Result<ClassifyResponse, FfiError> {
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        unsafe {
            apply_exec(&request.exec);
            let img = request
                .image
                .ok_or_else(|| err("missing image", INVALID_ARGUMENT))?;
            let t = match img.source {
                Some(image::Source::Tensor(t)) => t,
                _ => {
                    return Err(err(
                        "encoded-image decode is a TODO; pass Image.tensor (NHWC)",
                        UNIMPLEMENTED,
                    ))
                }
            };
            let in_name = if img.input_name.is_empty() {
                CStr::from_ptr(addr_of!(g_first_input) as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            } else {
                img.input_name.clone()
            };
            let cname = CString::new(in_name.as_str())
                .map_err(|_| err("bad input name", INVALID_ARGUMENT))?;
            let mut numel: c_long = 0;
            let dst = engine_input_ptr(cname.as_ptr(), &mut numel);
            if dst.is_null() {
                return Err(err(
                    format!("unknown input tensor: {in_name}"),
                    INVALID_ARGUMENT,
                ));
            }
            let mut input = t;
            input.name = in_name;
            copy_tensor_to_input(&input)?;
            g_last_token = -1;
            if engine_forward() != 0 {
                return Err(err("engine_forward failed", INTERNAL));
            }

            let (logits, ln) = match request.logits_tensor.as_ref().filter(|s| !s.is_empty()) {
                Some(name) => {
                    let cn = CString::new(name.as_str())
                        .map_err(|_| err("bad logits name", INVALID_ARGUMENT))?;
                    let mut ne: c_long = 0;
                    let p = engine_tensor_ptr(cn.as_ptr(), &mut ne);
                    if p.is_null() {
                        return Err(err("no such logits tensor", NOT_FOUND));
                    }
                    (p as *const f32, ne as usize)
                }
                None => {
                    let mut c: c_int = 0;
                    let p = engine_last_logits(&mut c);
                    if p.is_null() {
                        return Err(err("no logits to classify", INTERNAL));
                    }
                    (p, c as usize)
                }
            };
            let sl = std::slice::from_raw_parts(logits, ln);
            let k = (request.top_k.unwrap_or(5).max(0) as usize).min(ln);
            let mut idx: Vec<usize> = (0..ln).collect();
            idx.sort_by(|&a, &b| {
                sl[b]
                    .partial_cmp(&sl[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let classes = idx
                .into_iter()
                .take(k)
                .map(|i| ClassScore {
                    index: i as i32,
                    score: sl[i],
                    label: request.labels.get(i).cloned().unwrap_or_default(),
                })
                .collect();
            Ok(ClassifyResponse {
                classes,
                timing: None,
            })
        }
    }

    fn detect(&self, _request: DetectRequest) -> Result<DetectResponse, FfiError> {
        Err(err(
            "Detect not yet ported from native/main.c",
            UNIMPLEMENTED,
        ))
    }

    fn recognize_c_t_c(&self, _request: CTCRequest) -> Result<CTCResponse, FfiError> {
        Err(err(
            "RecognizeCTC not yet ported from native/main.c",
            UNIMPLEMENTED,
        ))
    }
}

// Register the plugin when the shared library is loaded, and also expose an
// explicit init for hosts that call one. register uses OnceLock::set, so the
// second call is a harmless no-op.
#[ctor::ctor]
fn on_load() {
    register_volvox_a_i_service_plugin(Plugin::new());
}

#[no_mangle]
pub extern "C" fn VolvoxAI_Init() {
    register_volvox_a_i_service_plugin(Plugin::new());
}
