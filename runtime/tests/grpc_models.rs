//! End-to-end test: drive the built plugin the "gRPC way" — through the exact
//! Synurang FFI entry point a remote/gRPC caller hits (`Synurang_Invoke_*` with
//! serialized protobuf), not by calling the engine directly.
//!
//! Loads each bundled model and runs one forward through the service:
//!   efficientdet_lite0_{fp32,fp16,int8}  -> Run with a zero image, read outputs
//!   tinystories_1m                       -> Run tokens/positions, check argmax
//!                                           matches the reference output.f32
//!
//! Run with:  cd runtime && cargo test --release -- --nocapture

use prost::Message;
use std::ffi::{c_char, c_int, c_void, CString};
use volvoxai::pb::{self, *};

extern "C" {
    fn Synurang_Invoke_VolvoxAIService(
        method: *const c_char,
        data: *const c_char,
        data_len: c_int,
        resp_len: *mut c_int,
    ) -> *mut c_char;
    fn Synurang_Free(ptr: *mut c_void);
}

// Mirrors the generated CoreFfiError payload, so failures print a real message.
#[derive(Clone, PartialEq, prost::Message)]
struct SvcError {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(int32, tag = "3")]
    grpc_code: i32,
}

// Unary call through the FFI dispatcher. resp_len < 0 signals an FfiError payload.
fn call<Req: Message, Resp: Message + Default>(method: &str, req: &Req) -> Result<Resp, String> {
    let data = req.encode_to_vec();
    let m = CString::new(method).unwrap();
    let mut resp_len: c_int = 0;
    let (ptr, bytes) = unsafe {
        let ptr = Synurang_Invoke_VolvoxAIService(
            m.as_ptr(),
            data.as_ptr() as *const c_char,
            data.len() as c_int,
            &mut resp_len,
        );
        if ptr.is_null() && resp_len != 0 {
            return Err("null response".into());
        }
        let n = resp_len.unsigned_abs() as usize;
        let bytes = if n == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(ptr as *const u8, n).to_vec()
        };
        if !ptr.is_null() {
            Synurang_Free(ptr as *mut c_void);
        }
        (ptr, bytes)
    };
    let _ = ptr;
    if resp_len < 0 {
        let e = SvcError::decode(&*bytes)
            .map(|e| e.message)
            .unwrap_or_default();
        return Err(format!("service error: {e}"));
    }
    Resp::decode(&*bytes).map_err(|e| format!("decode: {e}"))
}

// proto DType tag + byte width for a config dtype string.
fn dtype_of(s: &str) -> (i32, usize) {
    match s {
        "float32" => (1, 4),
        "float16" => (2, 2),
        "int32" => (10, 4),
        "uint8" => (16, 1),
        "int8" => (8, 1),
        other => panic!("unhandled input dtype: {other}"),
    }
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
        .unwrap()
}

fn run_model(name: &str) {
    let dir = format!("../models/{name}");
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).unwrap()).unwrap();

    // LoadModel
    let tok = format!("{dir}/vocab.bin");
    let tokenizer_path = if std::path::Path::new(&tok).exists() {
        tok
    } else {
        String::new()
    };
    let load: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAIService/LoadModel",
        &LoadModelRequest {
            source: Some(pb::load_model_request::Source::Paths(ModelPaths {
                config_path: format!("{dir}/config.json"),
                weights_path: format!("{dir}/model.safetensors"),
                tokenizer_path,
            })),
            exec: None,
            alias: None,
        },
    )
    .unwrap_or_else(|e| panic!("[{name}] LoadModel: {e}"));
    let info = load.info.unwrap_or_default();
    assert!(!load.model_id.is_empty(), "[{name}] empty model_id");
    assert!(info.num_ops > 0, "[{name}] num_ops == 0");
    println!(
        "[{name}] loaded {} — {} ops, tokenizer={}",
        load.model_id, info.num_ops, info.has_tokenizer
    );

    // Build one Tensor per declared input (example file if present, else zeros).
    let mut inputs = Vec::new();
    let mut last_token = None;
    for (iname, spec) in cfg["inputs"].as_object().unwrap() {
        let shape: Vec<i64> = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let numel: usize = shape.iter().product::<i64>() as usize;
        let (dtype, width) = dtype_of(spec["dtype"].as_str().unwrap());
        let ext = if width == 4 && dtype == 10 {
            "i32"
        } else if dtype <= 2 {
            "f32"
        } else {
            ""
        };
        let file = format!("{dir}/{iname}.{ext}");
        let data = match std::fs::read(&file) {
            Ok(d) if d.len() == numel * width => d,
            _ => vec![0u8; numel * width],
        };
        if iname == "tokens" && dtype == 10 {
            let pad = 50256;
            let token_count = data
                .chunks_exact(4)
                .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .position(|id| id == pad)
                .unwrap_or(numel);
            if token_count > 0 {
                last_token = Some(token_count as i32 - 1);
            }
        }
        inputs.push(Tensor {
            name: iname.clone(),
            shape,
            dtype,
            data,
            quant: None,
        });
    }

    let outputs: Vec<String> = cfg["outputs"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(name, value)| value.as_str().unwrap_or(name).to_string())
                .collect()
        })
        .unwrap_or_default();
    let is_lm = info.has_tokenizer;

    let resp: RunResponse = call(
        "/volvoxai.v1.VolvoxAIService/Run",
        &RunRequest {
            model_id: load.model_id.clone(),
            inputs,
            output_names: outputs.clone(),
            last_token: if is_lm { last_token } else { None },
            exec: None,
        },
    )
    .unwrap_or_else(|e| panic!("[{name}] Run: {e}"));

    assert!(!resp.outputs.is_empty(), "[{name}] no outputs");
    for t in &resp.outputs {
        assert_eq!(t.dtype, 1, "[{name}] output {} is not F32", t.name);
        let floats = f32_vec(&t.data);
        assert!(!floats.is_empty(), "[{name}] output {} empty", t.name);
        assert!(
            floats.iter().all(|x| x.is_finite()),
            "[{name}] output {} not finite",
            t.name
        );
        println!(
            "[{name}]   output {:<28} shape={:?} n={}",
            t.name,
            t.shape,
            floats.len()
        );
    }

    // tinystories: check the last-token argmax matches the reference output.f32.
    if is_lm {
        let reference = std::fs::read(format!("{dir}/output.f32")).unwrap();
        let refv = f32_vec(&reference);
        let got = f32_vec(&resp.outputs[0].data);
        assert_eq!(
            got.len(),
            refv.len(),
            "[{name}] logits len {} != ref {}",
            got.len(),
            refv.len()
        );
        let (a, b) = (argmax(&got), argmax(&refv));
        let max_diff = got
            .iter()
            .zip(&refv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!(
            "[{name}]   argmax got={a} ref={b} (vocab={}), max|Δ|={max_diff:.4}",
            got.len()
        );
        assert_eq!(a, b, "[{name}] argmax token mismatch vs reference");
    }

    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAIService/UnloadModel",
        &ModelRef {
            model_id: load.model_id,
        },
    )
    .unwrap_or_else(|e| panic!("[{name}] UnloadModel: {e}"));
}

// Decode a little-endian F32 tensor payload.
fn f32_vec(b: &[u8]) -> Vec<f32> {
    assert!(b.len() % 4 == 0);
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

#[test]
fn models_work_via_synurang_ffi_grpc_boundary() {
    volvoxai::VolvoxAI_Init(); // register the plugin

    let ping: PingResponse =
        call("/volvoxai.v1.VolvoxAIService/Ping", &pb::Empty {}).expect("Ping");
    println!("ping: {}", ping.version);

    for m in [
        "efficientdet_lite0_fp32",
        "efficientdet_lite0_fp16",
        "efficientdet_lite0_int8",
        "tinystories_1m",
    ] {
        run_model(m);
    }
}
