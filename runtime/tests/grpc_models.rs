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

fn string_entries<const N: usize>(entries: [(&str, &str); N]) -> Vec<StringEntry> {
    entries
        .into_iter()
        .map(|(key, value)| StringEntry {
            key: key.to_string(),
            value: value.to_string(),
        })
        .collect()
}

fn tensor_shape_entries<const N: usize>(
    entries: [(&str, Vec<i64>); N],
) -> Vec<TensorShapeEntry> {
    entries
        .into_iter()
        .map(|(key, dims)| TensorShapeEntry {
            key: key.to_string(),
            value: Some(TensorShape { dims }),
        })
        .collect()
}

fn string_entry_value<'a>(entries: &'a [StringEntry], key: &str) -> Option<&'a str> {
    entries
        .iter()
        .rev()
        .find(|entry| entry.key == key)
        .map(|entry| entry.value.as_str())
}

extern "C" {
    fn Synurang_Invoke_VolvoxAiService(
        method: *const c_char,
        data: *const c_char,
        data_len: c_int,
        resp_len: *mut c_int,
    ) -> *mut c_char;
    fn Synurang_Free(ptr: *mut c_void);
    fn Synurang_Stream_VolvoxAiService_Open(method: *const c_char) -> u64;
    fn Synurang_Stream_Send(handle: u64, data: *const c_char, data_len: c_int) -> c_int;
    fn Synurang_Stream_Recv(handle: u64, resp_len: *mut c_int, status: *mut c_int) -> *mut c_char;
    fn Synurang_Stream_CloseSend(handle: u64);
    fn Synurang_Stream_Close(handle: u64);
    fn vk_training_available() -> c_int;
    fn opengl_training_available() -> c_int;
    #[cfg(target_os = "macos")]
    fn metal_training_available() -> c_int;
    fn volvoxai_engine_last_training_backend() -> c_int;
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
fn call_status<Req: Message, Resp: Message + Default>(
    method: &str,
    req: &Req,
) -> Result<Resp, SvcError> {
    let data = req.encode_to_vec();
    let m = CString::new(method).unwrap();
    let mut resp_len: c_int = 0;
    let (ptr, bytes) = unsafe {
        let ptr = Synurang_Invoke_VolvoxAiService(
            m.as_ptr(),
            data.as_ptr() as *const c_char,
            data.len() as c_int,
            &mut resp_len,
        );
        if ptr.is_null() && resp_len != 0 {
            return Err(SvcError { code: 13, message: "null response".into(), grpc_code: 13 });
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
        return Err(SvcError::decode(&*bytes).unwrap_or(SvcError {
            code: 13,
            message: "invalid service error payload".into(),
            grpc_code: 13,
        }));
    }
    Resp::decode(&*bytes).map_err(|error| SvcError {
        code: 13,
        message: format!("decode: {error}"),
        grpc_code: 13,
    })
}

fn call<Req: Message, Resp: Message + Default>(method: &str, req: &Req) -> Result<Resp, String> {
    call_status(method, req).map_err(|error| format!("service error: {}", error.message))
}

fn generate_once(model_id: &str) -> Result<(), String> {
    let request = GenerateRequest {
        model_id: model_id.to_string(),
        prompt: "Once".to_string(),
        image: None,
        r#gen: Some(GenerationConfig {
            max_new_tokens: Some(1),
            ..Default::default()
        }),
        exec: None,
        adapters: None,
    }
    .encode_to_vec();
    let method = CString::new("/volvoxai.v1.VolvoxAiService/Generate").unwrap();
    unsafe {
        let handle = Synurang_Stream_VolvoxAiService_Open(method.as_ptr());
        if handle == 0 {
            return Err("open Generate stream failed".into());
        }
        if Synurang_Stream_Send(
            handle,
            request.as_ptr() as *const c_char,
            request.len() as c_int,
        ) != 0
        {
            Synurang_Stream_Close(handle);
            return Err("send Generate request failed".into());
        }
        Synurang_Stream_CloseSend(handle);
        let mut saw_done = false;
        loop {
            let mut len = 0;
            let mut status = 0;
            let ptr = Synurang_Stream_Recv(handle, &mut len, &mut status);
            if status == 1 {
                break;
            }
            let bytes = if ptr.is_null() || len <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(ptr as *const u8, len as usize).to_vec()
            };
            if !ptr.is_null() {
                Synurang_Free(ptr as *mut c_void);
            }
            if status < 0 {
                Synurang_Stream_Close(handle);
                let message = SvcError::decode(bytes.as_slice())
                    .map(|error| error.message)
                    .unwrap_or_else(|_| "unknown Generate error".to_string());
                return Err(message);
            }
            let event =
                GenerateEvent::decode(bytes.as_slice()).map_err(|error| error.to_string())?;
            saw_done |= matches!(event.event, Some(pb::generate_event::Event::Done(_)));
        }
        Synurang_Stream_Close(handle);
        if !saw_done {
            return Err("Generate stream omitted done event".into());
        }
    }
    Ok(())
}

// proto DataType tag + byte width for a config dtype string.
fn dtype_of(s: &str) -> (i32, usize) {
    match s {
        "float32" => (18, 4),
        "float16" => (14, 2),
        "int32" => (16, 4),
        "uint8" => (5, 1),
        "int8" => (6, 1),
        other => panic!("unhandled input dtype: {other}"),
    }
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
        .unwrap()
}

fn f32_tensor(name: &str, shape: Vec<i64>, value: f32) -> Tensor {
    let numel = shape.iter().product::<i64>() as usize;
    let mut data = Vec::with_capacity(numel * 4);
    for _ in 0..numel {
        data.extend_from_slice(&value.to_le_bytes());
    }
    Tensor {
        name: name.to_string(),
        shape,
        dtype: 18,
        data,
        quant: None,
        access_flags: 0,
        initializer: None,
    }
}

fn f32_values_tensor(name: &str, shape: Vec<i64>, values: &[f32]) -> Tensor {
    assert_eq!(shape.iter().product::<i64>() as usize, values.len());
    Tensor {
        name: name.to_string(),
        shape,
        dtype: 18,
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        quant: None,
        access_flags: 0,
        initializer: None,
    }
}

fn api_matmul_node() -> GraphNode {
    GraphNode {
        index: 0,
        id: String::new(),
        op: 1,
        op_name: String::new(), // enum-only CreateModel canonicalization
        inputs: string_entries([("input", "x"), ("weight", "w")]),
        outputs: string_entries([("out", "y")]),
        output_shapes: tensor_shape_entries([("out", vec![1, 2])]),
        params_json: None,
    }
}

fn api_model_request(with_node: bool) -> CreateModelRequest {
    CreateModelRequest {
        inputs: vec![TensorSpec {
            name: "x".to_string(),
            shape: vec![1, 2],
            dtype: 18,
            access_flags: 0,
            size_bytes: 8,
        }],
        graph: Some(GraphInfo {
            nodes: if with_node {
                vec![api_matmul_node()]
            } else {
                Vec::new()
            },
        }),
        tensors: vec![
            f32_values_tensor("w", vec![2, 2], &[2.0, 0.0, 0.0, 3.0]),
            Tensor {
                name: "step_i32".to_string(),
                shape: vec![1],
                dtype: 16,
                data: 0i32.to_le_bytes().to_vec(),
                quant: None,
                access_flags: 0,
                initializer: None,
            },
        ],
        output_names: if with_node {
            vec!["y".to_string()]
        } else {
            Vec::new()
        },
        exec: None,
        tokenizer: Vec::new(),
        metadata: Vec::new(),
    }
}

fn api_model_backing_dir(model_id: &str) -> std::path::PathBuf {
    let inspection: ModelInspection = call(
        "/volvoxai.v1.VolvoxAiService/InspectModel",
        &InspectModelRequest {
            model_id: model_id.to_string(),
            include_graph: false,
            include_tensors: false,
            include_weight_files: true,
            include_params_json: false,
        },
    )
    .expect("InspectModel for API backing");
    let path = &inspection
        .weight_files
        .first()
        .expect("API model weight file")
        .path;
    std::path::Path::new(path)
        .parent()
        .expect("API model backing parent")
        .to_path_buf()
}

fn run_api_model(model_id: &str, explicit_outputs: bool) -> Vec<f32> {
    let response: RunResponse = call(
        "/volvoxai.v1.VolvoxAiService/Run",
        &RunRequest {
            model_id: model_id.to_string(),
            inputs: vec![f32_values_tensor("x", vec![1, 2], &[1.0, 2.0])],
            output_names: if explicit_outputs {
                vec!["y".to_string()]
            } else {
                Vec::new()
            },
            last_token: None,
            exec: None,
            adapters: None,
        },
    )
    .expect("Run API-created model");
    assert_eq!(response.outputs.len(), 1);
    f32_vec(&response.outputs[0].data)
}

fn api_train_request(model_id: &str, trainable_tensors: Vec<String>) -> TrainStepRequest {
    TrainStepRequest {
        model_id: model_id.to_string(),
        inputs: vec![f32_values_tensor("x", vec![1, 2], &[1.0, 2.0])],
        logits_tensor: "y".to_string(),
        target_ids: vec![0],
        trainable_tensors,
        optimizer: Some(OptimizerOptions {
            learning_rate: 0.05,
            beta1: None,
            beta2: None,
            epsilon: None,
            weight_decay: None,
            max_grad_norm: None,
            step: Some(1),
        }),
        update_mode: Some(2),
        persist_safetensors: false,
        weight_paths: Vec::new(),
        last_token: None,
        ignore_id: None,
        losses: Vec::new(),
        accumulation: None,
    }
}

fn exercise_gpu_backend(backend: i32, name: &str, required_env: &str) {
    let mut request = api_model_request(true);
    request.exec = Some(ExecOptions { backend: Some(backend), ..Default::default() });
    let load: LoadModelResponse = match call_status(
        "/volvoxai.v1.VolvoxAiService/CreateModel", &request,
    ) {
        Ok(load) => load,
        Err(error) if error.grpc_code == 14 => {
            if std::env::var(required_env).as_deref() == Ok("1") {
                panic!("{name} is required but unavailable: {}", error.message);
            }
            println!("[{name}] unavailable; skipping hardware exercise: {}", error.message);
            return;
        }
        Err(error) => panic!(
            "{name} CreateModel failed with unexpected status {}: {}",
            error.grpc_code, error.message
        ),
    };

    let ping: PingResponse = call("/volvoxai.v1.VolvoxAiService/Ping", &pb::Empty {})
        .expect("Ping after GPU CreateModel");
    assert!(ping.available_backends.contains(&backend));
    let available = unsafe { gpu_training_available(backend) };
    assert_eq!(available, 1, "{name} was reported without an initialized backend");

    let model_id = load.model_id;
    let thread_model_id = model_id.clone();
    let output = std::thread::spawn(move || run_api_model(&thread_model_id, false))
        .join()
        .unwrap_or_else(|_| panic!("{name} Run panicked on a different worker thread"));
    assert_eq!(output, vec![2.0, 6.0]);

    let rejected_override: Result<RunResponse, SvcError> = call_status(
        "/volvoxai.v1.VolvoxAiService/Run",
        &RunRequest {
            model_id: model_id.clone(),
            inputs: vec![f32_values_tensor("x", vec![1, 2], &[1.0, 2.0])],
            output_names: Vec::new(),
            last_token: None,
            exec: Some(ExecOptions { backend: Some(1), ..Default::default() }),
            adapters: None,
        },
    );
    assert_eq!(rejected_override.unwrap_err().grpc_code, 9);

    let trained: TrainStepResponse = call(
        "/volvoxai.v1.VolvoxAiService/TrainStep",
        &api_train_request(&model_id, vec!["w".to_string()]),
    )
    .unwrap_or_else(|error| panic!("{name} TrainStep: {error}"));
    assert!(trained.update_applied);
    assert_eq!(trained.step, 1);
    assert_eq!(
        unsafe { volvoxai_engine_last_training_backend() },
        backend - 1,
        "{name} TrainStep silently fell back to CPU"
    );

    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel", &ModelRef { model_id },
    )
    .unwrap_or_else(|error| panic!("{name} UnloadModel: {error}"));
    let ping: PingResponse = call("/volvoxai.v1.VolvoxAiService/Ping", &pb::Empty {})
        .expect("Ping after GPU UnloadModel");
    assert!(!ping.available_backends.contains(&backend));
    let available = unsafe { gpu_training_available(backend) };
    assert_eq!(available, 0, "{name} resources remained initialized after unload");
    println!("[{name}] gRPC/FFI init, cross-thread Run, TrainStep, and cleanup passed");
}

unsafe fn gpu_training_available(backend: i32) -> c_int {
    match backend {
        2 => vk_training_available(),
        3 => opengl_training_available(),
        #[cfg(target_os = "macos")]
        4 => metal_training_available(),
        _ => 0,
    }
}

fn exercise_backend_rejection_preserves_model() {
    let loaded: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/CreateModel", &api_model_request(true),
    )
    .expect("create CPU model before backend rejection");
    let rejected_backends: &[(i32, i32)] = if cfg!(target_os = "macos") {
        &[(99, 3)]
    } else {
        &[(4, 12), (99, 3)]
    };
    for &(backend, expected_status) in rejected_backends {
        let mut replacement = api_model_request(true);
        replacement.exec = Some(ExecOptions { backend: Some(backend), ..Default::default() });
        let rejected: Result<LoadModelResponse, SvcError> = call_status(
            "/volvoxai.v1.VolvoxAiService/CreateModel", &replacement,
        );
        assert_eq!(rejected.unwrap_err().grpc_code, expected_status);
        assert_eq!(run_api_model(&loaded.model_id, false), vec![2.0, 6.0]);
    }
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel",
        &ModelRef { model_id: loaded.model_id },
    )
    .expect("unload preserved CPU model");
}

fn api_lora_manifest(a: &Tensor, b: &Tensor) -> AdapterManifest {
    let binding = |role, tensor: &Tensor| AdapterTensorBinding {
        role,
        tensor_name: tensor.name.clone(),
        spec: Some(TensorSpec {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            access_flags: 3,
            size_bytes: tensor.data.len() as i64,
        }),
    };
    AdapterManifest {
        format: "volvox.adapter.v1".to_string(),
        adapter_id: "api-lora".to_string(),
        kind: AdapterKind::AdapterLora as i32,
        targets: vec![AdapterTarget {
            target_id: "api-w".to_string(),
            op_id: String::new(),
            base_tensor: "w".to_string(),
            adapter_layout: AdapterMatrixLayout::Canonical as i32,
            rank: 1,
            alpha: 1.0,
            scale: None,
            tensors: vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, a),
                binding(AdapterTensorRole::AdapterTensorB as i32, b),
            ],
        }],
        metadata: Vec::new(),
    }
}

fn run_api_model_base_route(model_id: &str) -> Vec<f32> {
    let response: RunResponse = call(
        "/volvoxai.v1.VolvoxAiService/Run",
        &RunRequest {
            model_id: model_id.to_string(),
            inputs: vec![f32_values_tensor("x", vec![1, 2], &[1.0, 2.0])],
            output_names: Vec::new(),
            last_token: None,
            exec: None,
            adapters: Some(AdapterSelection::default()),
        },
    )
    .expect("Run API-created model on explicit base route");
    assert_eq!(response.outputs.len(), 1);
    f32_vec(&response.outputs[0].data)
}

fn exercise_create_model() {
    let empty: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/CreateModel",
        &api_model_request(false),
    )
    .expect("CreateModel with an empty graph");
    let empty_info = empty.info.as_ref().expect("CreateModel info");
    assert_eq!(empty_info.num_ops, 0);
    assert!(empty_info.writable_weights);
    assert!(empty_info.patchable_graph);
    let first_backing = api_model_backing_dir(&empty.model_id);
    assert!(first_backing.is_dir());

    let patched: PatchGraphResponse = call(
        "/volvoxai.v1.VolvoxAiService/PatchGraph",
        &PatchGraphRequest {
            model_id: empty.model_id.clone(),
            patches: vec![GraphNodePatch {
                node_index: Some(-1),
                mode: Some(4), // NODE_PATCH_MODE_INSERT_AFTER: append / first node
                op: Some(1),   // OP_MATMUL; enum-only canonicalization
                op_name: None,
                inputs: api_matmul_node().inputs,
                outputs: api_matmul_node().outputs,
                output_shapes: api_matmul_node().output_shapes,
                params_json: None,
            }],
            rebuild_plan: Some(true),
            reoptimize: Some(true),
            persist_config: true,
            declared_outputs: Some(DeclaredOutputsUpdate {
                names: vec!["y".to_string()],
            }),
        },
    )
    .expect("append first API model node");
    let patched_info = patched.info.as_ref().expect("patched model info");
    assert_eq!(patched_info.num_ops, 1);
    assert_eq!(patched_info.outputs[0].name, "y");
    assert_eq!(run_api_model(&empty.model_id, false), vec![2.0, 6.0]);

    let missing_declared_output = call::<_, PatchGraphResponse>(
        "/volvoxai.v1.VolvoxAiService/PatchGraph",
        &PatchGraphRequest {
            model_id: empty.model_id.clone(),
            patches: Vec::new(),
            rebuild_plan: None,
            reoptimize: None,
            persist_config: false,
            declared_outputs: Some(DeclaredOutputsUpdate {
                names: vec!["missing".to_string()],
            }),
        },
    );
    assert!(
        missing_declared_output.is_err(),
        "accepted a declared output that the graph does not produce"
    );
    assert_eq!(run_api_model(&empty.model_id, false), vec![2.0, 6.0]);

    let reloaded: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/LoadModel",
        &LoadModelRequest {
            source: Some(load_model_request::Source::Paths(ModelPaths {
                config_path: first_backing.join("config.json").display().to_string(),
                weights_path: first_backing
                    .join("model.safetensors")
                    .display()
                    .to_string(),
                tokenizer_path: String::new(),
                extra_weights_paths: Vec::new(),
            })),
            exec: None,
            alias: None,
            edit: Some(ModelEditOptions {
                open_weights_writable: true,
                allow_tensor_updates: true,
                allow_graph_patches: true,
            }),
        },
    )
    .expect("reload persisted API graph/default outputs");
    assert_eq!(
        reloaded.info.as_ref().unwrap().outputs[0].name,
        "y",
        "persisted declared output was not restored"
    );
    let model_id = reloaded.model_id;
    assert_eq!(run_api_model(&model_id, false), vec![2.0, 6.0]);

    let conflicting_op = call::<_, PatchGraphResponse>(
        "/volvoxai.v1.VolvoxAiService/PatchGraph",
        &PatchGraphRequest {
            model_id: model_id.clone(),
            patches: vec![GraphNodePatch {
                node_index: Some(0),
                mode: Some(1),
                op: Some(80), // OP_ADD
                op_name: Some("MatMul".to_string()),
                inputs: Vec::new(),
                outputs: Vec::new(),
                output_shapes: Vec::new(),
                params_json: None,
            }],
            rebuild_plan: None,
            reoptimize: None,
            persist_config: false,
            declared_outputs: None,
        },
    );
    assert!(conflicting_op.is_err(), "accepted conflicting op/op_name");

    let failed_batch = call::<_, PatchGraphResponse>(
        "/volvoxai.v1.VolvoxAiService/PatchGraph",
        &PatchGraphRequest {
            model_id: model_id.clone(),
            patches: vec![
                GraphNodePatch {
                    node_index: Some(0),
                    mode: Some(1),
                    op: None,
                    op_name: None,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    output_shapes: Vec::new(),
                    params_json: Some("{\"should_not_commit\":true}".to_string()),
                },
                GraphNodePatch {
                    node_index: Some(99),
                    mode: Some(1),
                    op: None,
                    op_name: None,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    output_shapes: Vec::new(),
                    params_json: Some("{}".to_string()),
                },
            ],
            rebuild_plan: None,
            reoptimize: None,
            persist_config: false,
            declared_outputs: None,
        },
    );
    assert!(failed_batch.is_err(), "accepted invalid multi-patch batch");
    assert_eq!(run_api_model(&model_id, false), vec![2.0, 6.0]);

    let before_failed_updates: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("weight before invalid update batch");
    let invalid_updates = call::<_, ApplyTensorUpdatesResponse>(
        "/volvoxai.v1.VolvoxAiService/ApplyTensorUpdates",
        &ApplyTensorUpdatesRequest {
            model_id: model_id.clone(),
            updates: vec![
                TensorUpdate {
                    tensor: Some(f32_values_tensor("w", vec![2, 2], &[1.0, 0.0, 0.0, 0.0])),
                    mode: TensorUpdateMode::TensorUpdateAdd as i32,
                    optimizer: None,
                },
                TensorUpdate {
                    tensor: Some(f32_values_tensor("missing", vec![1], &[f32::NAN])),
                    mode: TensorUpdateMode::TensorUpdateAdamw as i32,
                    optimizer: None,
                },
            ],
            optimizer: None,
            persist_safetensors: false,
            weight_paths: Vec::new(),
        },
    );
    assert!(invalid_updates.is_err(), "accepted invalid tensor update batch");
    let after_failed_updates: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("weight after invalid update batch");
    assert_eq!(before_failed_updates.data, after_failed_updates.data);

    let input_optimizer_update = call::<_, ApplyTensorUpdatesResponse>(
        "/volvoxai.v1.VolvoxAiService/ApplyTensorUpdates",
        &ApplyTensorUpdatesRequest {
            model_id: model_id.clone(),
            updates: vec![TensorUpdate {
                tensor: Some(f32_values_tensor("x", vec![1, 2], &[0.25, -0.25])),
                mode: TensorUpdateMode::TensorUpdateAdamw as i32,
                optimizer: None,
            }],
            optimizer: None,
            persist_safetensors: false,
            weight_paths: Vec::new(),
        },
    );
    assert!(
        input_optimizer_update.is_err(),
        "ApplyTensorUpdates accepted a graph input as an optimizer target"
    );
    let input_assign = call::<_, ApplyTensorUpdatesResponse>(
        "/volvoxai.v1.VolvoxAiService/ApplyTensorUpdates",
        &ApplyTensorUpdatesRequest {
            model_id: model_id.clone(),
            updates: vec![TensorUpdate {
                tensor: Some(f32_values_tensor("x", vec![1, 2], &[0.25, -0.25])),
                mode: TensorUpdateMode::TensorUpdateAssign as i32,
                optimizer: None,
            }],
            optimizer: None,
            persist_safetensors: false,
            weight_paths: Vec::new(),
        },
    );
    assert!(
        input_assign.is_err(),
        "ApplyTensorUpdates ASSIGN accepted a graph input instead of the input API"
    );

    let update: ApplyTensorUpdatesResponse = call(
        "/volvoxai.v1.VolvoxAiService/ApplyTensorUpdates",
        &ApplyTensorUpdatesRequest {
            model_id: model_id.clone(),
            updates: vec![TensorUpdate {
                tensor: Some(f32_values_tensor("w", vec![2, 2], &[1.0, 0.0, 0.0, 0.0])),
                mode: TensorUpdateMode::TensorUpdateAdd as i32,
                optimizer: None,
            }],
            optimizer: None,
            persist_safetensors: false,
            weight_paths: Vec::new(),
        },
    )
    .expect("train/update API-created model");
    assert_eq!(update.updated_tensors.len(), 1);
    assert_eq!(update.step, 0, "ADD-only edit advanced optimizer step");
    assert_eq!(run_api_model(&model_id, true), vec![3.0, 6.0]);

    let adapter_a = f32_tensor("api.w.lora_a", vec![2, 1], 0.0);
    let adapter_b = f32_tensor("api.w.lora_b", vec![1, 2], 0.0);
    let active_adapter: AdapterInfo = call(
        "/volvoxai.v1.VolvoxAiService/StageAdapter",
        &StageAdapterRequest {
            model_id: model_id.clone(),
            manifest: Some(api_lora_manifest(&adapter_a, &adapter_b)),
            tensors: vec![adapter_a, adapter_b],
            activate: true,
        },
    )
    .expect("stage active API-model adapter");
    let active_adapter_ref = active_adapter.adapter.clone().unwrap();
    assert!(active_adapter.active);

    let duplicate_trainables = call::<_, TrainStepResponse>(
        "/volvoxai.v1.VolvoxAiService/TrainStep",
        &api_train_request(&model_id, vec!["w".to_string(), "w".to_string()]),
    );
    assert!(
        duplicate_trainables.is_err(),
        "TrainStep accepted duplicate trainable tensors"
    );
    let integer_trainable = call::<_, TrainStepResponse>(
        "/volvoxai.v1.VolvoxAiService/TrainStep",
        &api_train_request(&model_id, vec!["step_i32".to_string()]),
    );
    assert!(
        integer_trainable.is_err(),
        "TrainStep accepted a non-F32 trainable tensor"
    );
    let input_trainable = call::<_, TrainStepResponse>(
        "/volvoxai.v1.VolvoxAiService/TrainStep",
        &api_train_request(&model_id, vec!["x".to_string()]),
    );
    assert!(
        input_trainable.is_err(),
        "TrainStep accepted a graph input as a checkpointed trainable"
    );
    let mut failed_native_train = api_train_request(&model_id, vec!["w".to_string()]);
    failed_native_train.target_ids = vec![99];
    failed_native_train.optimizer.as_mut().unwrap().step = None;
    assert!(
        call::<_, TrainStepResponse>(
            "/volvoxai.v1.VolvoxAiService/TrainStep",
            &failed_native_train,
        )
        .is_err(),
        "TrainStep accepted an out-of-range target"
    );
    assert_eq!(run_api_model(&model_id, false), vec![3.0, 6.0]);

    let mut valid_train = api_train_request(&model_id, vec!["w".to_string()]);
    valid_train.optimizer.as_mut().unwrap().step = None;
    let train: TrainStepResponse = call("/volvoxai.v1.VolvoxAiService/TrainStep", &valid_train)
        .expect("TrainStep on API-created base weight");
    assert!(train.loss.is_finite() && train.loss > 0.0);
    assert_eq!(train.examples, 1);
    assert_eq!(train.updated_tensors.len(), 1);
    assert_eq!(
        train.step, 1,
        "failed TrainStep consumed an implicit optimizer step"
    );
    let trained = run_api_model(&model_id, true);
    assert_eq!(
        trained,
        run_api_model_base_route(&model_id),
        "active staged adapter changed the TrainStep/base-route result"
    );
    assert!(trained[0] > 3.0 && trained[1] < 6.0);
    let stale_step = call::<_, TrainStepResponse>(
        "/volvoxai.v1.VolvoxAiService/TrainStep",
        &api_train_request(&model_id, vec!["w".to_string()]),
    );
    assert!(stale_step.is_err(), "TrainStep accepted a reused explicit optimizer step");

    let adapters_after_train: ListAdaptersResponse = call(
        "/volvoxai.v1.VolvoxAiService/ListAdapters",
        &ListAdaptersRequest {
            model_id: model_id.clone(),
            adapter_id: Some(active_adapter_ref.adapter_id.clone()),
        },
    )
    .expect("list active adapter after TrainStep");
    assert_eq!(adapters_after_train.adapters.len(), 1);
    let adapter_after_train = &adapters_after_train.adapters[0];
    assert_eq!(adapter_after_train.adapter, active_adapter.adapter);
    assert_eq!(adapter_after_train.manifest, active_adapter.manifest);
    assert_eq!(
        adapter_after_train.optimizer_step,
        active_adapter.optimizer_step
    );
    assert!(
        adapter_after_train.active,
        "TrainStep deactivated the adapter"
    );

    let legacy: LoRaTrainStepResponse = call(
        "/volvoxai.v1.VolvoxAiService/TrainLoRaStep",
        &LoRaTrainStepRequest {
            model_id: model_id.clone(),
            inputs: vec![f32_values_tensor("x", vec![1, 2], &[1.0, 2.0])],
            logits_tensor: "y".to_string(),
            target_ids: vec![0],
            trainable_tensors: vec!["w".to_string()],
            optimizer: Some(OptimizerOptions {
                learning_rate: 0.01,
                beta1: None,
                beta2: None,
                epsilon: None,
                weight_decay: None,
                max_grad_norm: None,
                step: Some(2),
            }),
            update_mode: Some(2),
            persist_safetensors: false,
            weight_paths: Vec::new(),
            last_token: None,
            ignore_id: None,
        },
    )
    .expect("legacy TrainLoRaStep delegates to TrainStep");
    assert!(legacy.loss.is_finite());
    assert_eq!(legacy.examples, 1);

    let _: AdapterState = call(
        "/volvoxai.v1.VolvoxAiService/ActivateAdapter",
        &ActivateAdapterRequest {
            model_id: model_id.clone(),
            selection: Some(AdapterSelection::default()),
        },
    )
    .expect("deactivate API-model adapter");
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/RemoveAdapter",
        &RemoveAdapterRequest {
            model_id: model_id.clone(),
            adapter: Some(active_adapter_ref),
        },
    )
    .expect("remove API-model adapter before graph edit");

    let deleted: PatchGraphResponse = call(
        "/volvoxai.v1.VolvoxAiService/PatchGraph",
        &PatchGraphRequest {
            model_id: model_id.clone(),
            patches: vec![GraphNodePatch {
                node_index: Some(0),
                mode: Some(5), // NODE_PATCH_MODE_DELETE
                op: None,
                op_name: None,
                inputs: Vec::new(),
                outputs: Vec::new(),
                output_shapes: Vec::new(),
                params_json: None,
            }],
            rebuild_plan: Some(false), // structural delete still forces rebuild
            reoptimize: None,
            persist_config: true,
            declared_outputs: Some(DeclaredOutputsUpdate { names: Vec::new() }),
        },
    )
    .expect("clear declared outputs and delete API model node");
    let deleted_info = deleted.info.expect("deleted model info");
    assert!(deleted_info.outputs.is_empty());
    assert!(deleted.changed_tensor_names.contains(&"y".to_string()));

    let complete: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/CreateModel",
        &api_model_request(true),
    )
    .expect("replace with complete API model");
    assert!(
        !first_backing.exists(),
        "CreateModel replacement leaked the previous private backing"
    );
    let second_backing = api_model_backing_dir(&complete.model_id);
    assert_eq!(run_api_model(&complete.model_id, false), vec![2.0, 6.0]);
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel",
        &ModelRef {
            model_id: complete.model_id,
        },
    )
    .expect("Unload API-created model");
    assert!(
        !second_backing.exists(),
        "UnloadModel leaked private API model backing"
    );
    println!("[CreateModel] append/default output/TrainStep/delete/cleanup passed");
}

fn exercise_multi_loss_gradient_accumulation() {
    let created: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/CreateModel",
        &api_model_request(true),
    )
    .expect("CreateModel for multi-loss accumulation");
    let before = run_api_model(&created.model_id, true);
    let request = TrainStepRequest {
        model_id: created.model_id.clone(),
        inputs: vec![f32_values_tensor("x", vec![1, 2], &[1.0, 2.0])],
        trainable_tensors: vec!["w".to_string()],
        optimizer: Some(OptimizerOptions {
            learning_rate: 0.05,
            beta1: None,
            beta2: None,
            epsilon: None,
            weight_decay: None,
            max_grad_norm: None,
            step: Some(1),
        }),
        update_mode: Some(2),
        losses: vec![
            CrossEntropyLoss {
                name: "tokens".to_string(),
                logits_tensor: "y".to_string(),
                target_ids: vec![0],
                weight: Some(1.0),
                ignore_id: None,
                last_token: None,
                normalizer: Some(2.0),
            },
            CrossEntropyLoss {
                name: "router".to_string(),
                logits_tensor: "y".to_string(),
                target_ids: vec![0],
                weight: Some(0.1),
                ignore_id: None,
                last_token: None,
                normalizer: Some(2.0),
            },
        ],
        accumulation: Some(GradientAccumulationOptions {
            steps: 2,
            flush: false,
            reset: false,
        }),
        ..Default::default()
    };

    let pending: TrainStepResponse = call(
        "/volvoxai.v1.VolvoxAiService/TrainStep",
        &request,
    )
    .expect("first accumulated multi-loss microbatch");
    assert!(!pending.update_applied);
    assert_eq!(pending.step, 0);
    assert_eq!(pending.accumulated_microbatches, 1);
    assert_eq!(pending.accumulation_steps, 2);
    assert!(pending.updated_tensors.is_empty());
    assert_eq!(pending.losses.len(), 2);
    assert_eq!(pending.losses[0].name, "tokens");
    assert_eq!(pending.losses[1].name, "router");
    assert_eq!(run_api_model(&created.model_id, true), before);

    let checkpoint_dir = std::env::temp_dir().join(format!(
        "volvox-pending-accumulation-checkpoint-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&checkpoint_dir);
    let checkpoint = call::<_, TrainingCheckpointInfo>(
        "/volvoxai.v1.VolvoxAiService/SaveTrainingCheckpoint",
        &SaveTrainingCheckpointRequest {
            model_id: created.model_id.clone(),
            directory: checkpoint_dir.to_string_lossy().into_owned(),
            overwrite: false,
            tokenizer: None,
            metadata: Vec::new(),
        },
    );
    assert!(
        checkpoint
            .unwrap_err()
            .contains("pending accumulated microbatch"),
        "checkpoint accepted an in-progress accumulation window"
    );

    let applied: TrainStepResponse = call(
        "/volvoxai.v1.VolvoxAiService/TrainStep",
        &request,
    )
    .expect("final accumulated multi-loss microbatch");
    assert!(applied.update_applied);
    assert_eq!(applied.step, 1);
    assert_eq!(applied.accumulated_microbatches, 2);
    assert_eq!(applied.accumulation_steps, 2);
    assert_eq!(applied.updated_tensors.len(), 1);
    assert_eq!(applied.losses.len(), 2);
    assert_eq!(applied.losses[0].examples, 2);
    assert_eq!(applied.losses[1].examples, 2);
    assert_ne!(run_api_model(&created.model_id, true), before);

    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel",
        &ModelRef {
            model_id: created.model_id,
        },
    )
    .expect("unload multi-loss accumulation model");
}

fn exercise_from_scratch_training_checkpoint() {
    let created: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/CreateModel",
        &CreateModelRequest {
            inputs: vec![TensorSpec {
                name: "x".to_string(),
                shape: vec![1, 2],
                dtype: 18,
                access_flags: 0,
                size_bytes: 8,
            }],
            graph: Some(GraphInfo { nodes: Vec::new() }),
            tensors: Vec::new(),
            output_names: Vec::new(),
            exec: None,
            tokenizer: Vec::new(),
            metadata: string_entries([("test", "from-scratch")]),
        },
    )
    .expect("CreateModel with zero initial tensors");
    let initializer = TensorInitializer {
        kind: 5, // XAVIER_NORMAL
        seed: Some(0x5eed),
        mean: None,
        stddev: None,
        gain: Some(1.0),
    };
    let add_weight = |model_id: &str| -> TensorSpec {
        call(
            "/volvoxai.v1.VolvoxAiService/AddModelTensor",
            &AddModelTensorRequest {
                model_id: model_id.to_string(),
                tensor: Some(Tensor {
                    name: "w".to_string(),
                    shape: vec![2, 2],
                    dtype: 18,
                    data: Vec::new(),
                    quant: None,
                    access_flags: 0,
                    initializer: Some(initializer.clone()),
                }),
            },
        )
        .expect("AddModelTensor initialized parameter")
    };
    let spec = add_weight(&created.model_id);
    assert_eq!(spec.shape, vec![2, 2]);
    let first_initialized: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: created.model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("read initialized parameter");
    assert!(f32_vec(&first_initialized.data).iter().all(|value| value.is_finite()));
    let golden_bits = [0xbe07_a013u32, 0x3f11_ebb3, 0x3e36_707a, 0x3e07_a836];
    let golden_bytes: Vec<u8> = golden_bits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(
        first_initialized.data, golden_bytes,
        "CreateModel initializer diverged from ts/training/Initializers.ts Mulberry32 golden vector"
    );
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/RemoveModelTensor",
        &RemoveModelTensorRequest {
            model_id: created.model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("remove unreferenced initialized parameter");
    add_weight(&created.model_id);
    let second_initialized: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: created.model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("read deterministically reinitialized parameter");
    assert_eq!(first_initialized.data, second_initialized.data);

    let _: PatchGraphResponse = call(
        "/volvoxai.v1.VolvoxAiService/PatchGraph",
        &PatchGraphRequest {
            model_id: created.model_id.clone(),
            patches: vec![GraphNodePatch {
                node_index: Some(-1),
                mode: Some(4),
                op: Some(1),
                op_name: None,
                inputs: string_entries([("input", "x"), ("weight", "w")]),
                outputs: string_entries([("out", "logits")]),
                output_shapes: tensor_shape_entries([("out", vec![1, 2])]),
                params_json: None,
            }],
            rebuild_plan: Some(true),
            reoptimize: Some(true),
            persist_config: true,
            declared_outputs: Some(DeclaredOutputsUpdate {
                names: vec!["logits".to_string()],
            }),
        },
    )
    .expect("patch first node after adding parameter");
    let train = |model_id: &str, include_optimizer: bool| -> TrainStepResponse {
        call(
            "/volvoxai.v1.VolvoxAiService/TrainStep",
            &TrainStepRequest {
                model_id: model_id.to_string(),
                inputs: vec![f32_values_tensor("x", vec![1, 2], &[1.0, -0.5])],
                logits_tensor: "logits".to_string(),
                target_ids: vec![0],
                trainable_tensors: vec!["w".to_string()],
                optimizer: include_optimizer.then_some(OptimizerOptions {
                    learning_rate: 0.01,
                    beta1: None,
                    beta2: None,
                    epsilon: None,
                    weight_decay: Some(0.01),
                    max_grad_norm: None,
                    step: None,
                }),
                update_mode: include_optimizer.then_some(3), // checkpoint restores this default
                persist_safetensors: false,
                weight_paths: Vec::new(),
                last_token: None,
                ignore_id: None,
                losses: Vec::new(),
                accumulation: None,
            },
        )
        .expect("from-scratch TrainStep")
    };
    assert_eq!(train(&created.model_id, true).step, 1);
    let checkpoint_dir = std::env::temp_dir().join(format!(
        "volvox-training-checkpoint-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&checkpoint_dir);
    let saved: TrainingCheckpointInfo = call(
        "/volvoxai.v1.VolvoxAiService/SaveTrainingCheckpoint",
        &SaveTrainingCheckpointRequest {
            model_id: created.model_id.clone(),
            directory: checkpoint_dir.to_string_lossy().into_owned(),
            overwrite: false,
            tokenizer: None,
            metadata: string_entries([("epoch", "1")]),
        },
    )
    .expect("SaveTrainingCheckpoint");
    assert_eq!(saved.training_step, 1);
    assert_eq!(saved.optimizer_file.as_ref().unwrap().tensors.len(), 2);
    assert_eq!(string_entry_value(&saved.metadata, "test"), Some("from-scratch"));
    assert_eq!(string_entry_value(&saved.metadata, "epoch"), Some("1"));

    let bad_checkpoint = checkpoint_dir.with_extension("bad");
    let _ = std::fs::remove_dir_all(&bad_checkpoint);
    std::fs::create_dir(&bad_checkpoint).expect("create invalid checkpoint copy");
    for entry in std::fs::read_dir(&checkpoint_dir).expect("list checkpoint") {
        let entry = entry.expect("checkpoint entry");
        std::fs::copy(entry.path(), bad_checkpoint.join(entry.file_name()))
            .expect("copy checkpoint entry");
    }
    let bad_config_path = bad_checkpoint.join("config.json");
    let mut bad_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&bad_config_path).expect("read copied config"),
    )
    .expect("parse copied config");
    bad_config["inputs"]["x"]["dtype"] = serde_json::json!("bool");
    std::fs::write(
        &bad_config_path,
        serde_json::to_vec_pretty(&bad_config).unwrap(),
    )
    .expect("write invalid checkpoint config");
    let before_failed_load: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: created.model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("weight before failed checkpoint load");
    let failed_load = call::<_, LoadTrainingCheckpointResponse>(
        "/volvoxai.v1.VolvoxAiService/LoadTrainingCheckpoint",
        &LoadTrainingCheckpointRequest {
            directory: bad_checkpoint.to_string_lossy().into_owned(),
            exec: None,
        },
    );
    assert!(failed_load.is_err(), "loaded checkpoint with unsupported input dtype");
    let after_failed_load: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: created.model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("existing handle was not restored after failed checkpoint load");
    assert_eq!(before_failed_load.data, after_failed_load.data);
    std::fs::remove_dir_all(&bad_checkpoint).expect("remove invalid checkpoint copy");

    let uninterrupted = train(&created.model_id, true);
    assert_eq!(uninterrupted.step, 2);
    let uninterrupted_weight: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: created.model_id,
            name: "w".to_string(),
        },
    )
    .expect("read uninterrupted second-step parameter");

    let loaded: LoadTrainingCheckpointResponse = call(
        "/volvoxai.v1.VolvoxAiService/LoadTrainingCheckpoint",
        &LoadTrainingCheckpointRequest {
            directory: checkpoint_dir.to_string_lossy().into_owned(),
            exec: None,
        },
    )
    .expect("LoadTrainingCheckpoint");
    let resumed_model = loaded.model.expect("checkpoint model");
    assert_eq!(loaded.checkpoint.as_ref().unwrap().training_step, 1);
    let resumed = train(&resumed_model.model_id, false);
    assert_eq!(resumed.step, 2);
    let resumed_weight: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: resumed_model.model_id.clone(),
            name: "w".to_string(),
        },
    )
    .expect("read resumed second-step parameter");
    assert_eq!(
        resumed_weight.data, uninterrupted_weight.data,
        "checkpoint resume did not restore AdamW moments/step exactly"
    );
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel",
        &ModelRef {
            model_id: resumed_model.model_id,
        },
    )
    .expect("unload resumed training model");
    std::fs::remove_dir_all(&checkpoint_dir).expect("remove training checkpoint");
    println!("[TrainingCheckpoint] zero-weight create/add/train/save/resume passed");
}

fn exercise_seq2seq_train_step() {
    let input = |name: &str, shape: Vec<i64>| TensorSpec {
        name: name.to_string(),
        size_bytes: shape.iter().product::<i64>() * 4,
        shape,
        dtype: 16,
        access_flags: 0,
    };
    let inputs = vec![
        input("encoder_tokens", vec![1, 3]),
        input("decoder_tokens", vec![1, 2]),
        input("encoder_mask", vec![1, 3]),
        input("decoder_mask", vec![1, 2]),
        input("encoder_positions", vec![1, 3]),
        input("decoder_positions", vec![1, 2]),
    ];
    let created: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/CreateModel",
        &CreateModelRequest {
            inputs,
            graph: Some(GraphInfo {
                nodes: vec![GraphNode {
                    index: 0,
                    id: "decoder_embedding".to_string(),
                    op: 30,
                    op_name: String::new(),
                    inputs: string_entries([
                        ("input", "decoder_tokens"),
                        ("weight", "embedding"),
                    ]),
                    outputs: string_entries([("out", "logits")]),
                    output_shapes: tensor_shape_entries([("out", vec![1, 2, 8])]),
                    params_json: None,
                }],
            }),
            tensors: vec![Tensor {
                name: "embedding".to_string(),
                shape: vec![8, 8],
                dtype: 18,
                data: Vec::new(),
                quant: None,
                access_flags: 0,
                initializer: Some(TensorInitializer {
                    kind: 4,
                    seed: Some(11),
                    mean: None,
                    stddev: None,
                    gain: Some(1.0),
                }),
            }],
            output_names: vec!["logits".to_string()],
            exec: None,
            tokenizer: Vec::new(),
            metadata: Vec::new(),
        },
    )
    .expect("CreateModel seq2seq helper graph");
    let request = |target_ids: Vec<i32>| Seq2SeqTrainStepRequest {
        model_id: created.model_id.clone(),
        source_ids: vec![4, 0, 5],
        target_ids,
        batch_size: 1,
        source_length: 3,
        target_length: 2,
        encoder_tokens_input: "encoder_tokens".to_string(),
        decoder_tokens_input: "decoder_tokens".to_string(),
        encoder_mask_input: Some("encoder_mask".to_string()),
        decoder_mask_input: Some("decoder_mask".to_string()),
        bos_id: 1,
        pad_id: 0,
        ignore_id: Some(-100),
        logits_tensor: "logits".to_string(),
        trainable_tensors: vec!["embedding".to_string()],
        optimizer: Some(OptimizerOptions {
            learning_rate: 0.01,
            beta1: None,
            beta2: None,
            epsilon: None,
            weight_decay: None,
            max_grad_norm: None,
            step: None,
        }),
        update_mode: Some(3),
        persist_safetensors: false,
        weight_paths: Vec::new(),
        encoder_positions_input: Some("encoder_positions".to_string()),
        decoder_positions_input: Some("decoder_positions".to_string()),
    };
    let trained: TrainStepResponse = call(
        "/volvoxai.v1.VolvoxAiService/Seq2SeqTrainStep",
        &request(vec![3, 0]),
    )
    .expect("Seq2SeqTrainStep teacher forcing");
    assert_eq!(trained.examples, 1);
    assert_eq!(trained.step, 1);
    let read_i32 = |name: &str| -> Vec<i32> {
        let tensor: Tensor = call(
            "/volvoxai.v1.VolvoxAiService/GetTensor",
            &GetTensorRequest {
                model_id: created.model_id.clone(),
                name: name.to_string(),
            },
        )
        .unwrap_or_else(|error| panic!("read seq2seq helper input {name}: {error}"));
        assert_eq!(tensor.dtype, 16);
        i32_vec(&tensor.data)
    };
    assert_eq!(read_i32("encoder_tokens"), vec![4, 0, 5]);
    assert_eq!(read_i32("decoder_tokens"), vec![1, 3]);
    assert_eq!(read_i32("encoder_mask"), vec![1, 0, 1]);
    assert_eq!(read_i32("decoder_mask"), vec![1, 1]);
    assert_eq!(read_i32("encoder_positions"), vec![0, 1, 2]);
    assert_eq!(read_i32("decoder_positions"), vec![0, 1]);
    let before_ignored: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: created.model_id.clone(),
            name: "embedding".to_string(),
        },
    )
    .expect("embedding before all-ignore step");
    let ignored: TrainStepResponse = call(
        "/volvoxai.v1.VolvoxAiService/Seq2SeqTrainStep",
        &request(vec![0, 0]),
    )
    .expect("all-ignore Seq2SeqTrainStep");
    assert_eq!(ignored.examples, 0);
    assert_eq!(ignored.step, 1, "all-ignore step advanced runtime training_step");
    let after_ignored: Tensor = call(
        "/volvoxai.v1.VolvoxAiService/GetTensor",
        &GetTensorRequest {
            model_id: created.model_id.clone(),
            name: "embedding".to_string(),
        },
    )
    .expect("embedding after all-ignore step");
    assert_eq!(before_ignored.data, after_ignored.data);
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel",
        &ModelRef {
            model_id: created.model_id,
        },
    )
    .expect("unload seq2seq helper graph");
    println!("[Seq2SeqTrainStep] shift/mask/position/ignore semantics passed");
}

fn lora_manifest(a: &Tensor, b: &Tensor) -> AdapterManifest {
    let binding = |role, tensor: &Tensor| AdapterTensorBinding {
        role,
        tensor_name: tensor.name.clone(),
        spec: Some(TensorSpec {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            access_flags: 3,
            size_bytes: tensor.data.len() as i64,
        }),
    };
    AdapterManifest {
        format: "volvox.adapter.v1".into(),
        adapter_id: "grpc-lora".into(),
        kind: AdapterKind::AdapterLora as i32,
        targets: vec![AdapterTarget {
            target_id: "h0-qkv".into(),
            op_id: String::new(),
            base_tensor: "h.0.attn.qkv_proj.weight".into(),
            adapter_layout: AdapterMatrixLayout::Canonical as i32,
            rank: 1,
            alpha: 1.0,
            scale: None,
            tensors: vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, a),
                binding(AdapterTensorRole::AdapterTensorB as i32, b),
            ],
        }],
        metadata: string_entries([("purpose", "grpc-roundtrip")]),
    }
}

fn run_lm_with_route(
    model_id: &str,
    inputs: &[Tensor],
    last_token: Option<i32>,
    adapters: Option<AdapterSelection>,
) -> RunResponse {
    call(
        "/volvoxai.v1.VolvoxAiService/Run",
        &RunRequest {
            model_id: model_id.to_string(),
            inputs: inputs.to_vec(),
            output_names: vec![],
            last_token,
            exec: None,
            adapters,
        },
    )
    .expect("adapter-routed Run")
}

fn exercise_adapter_lifecycle(
    model_id: &str,
    inputs: &[Tensor],
    last_token: Option<i32>,
    base: &RunResponse,
    model_dir: &str,
    tokenizer_path: &str,
) {
    let failed_reload = call::<_, LoadModelResponse>(
        "/volvoxai.v1.VolvoxAiService/LoadModel",
        &LoadModelRequest {
            source: Some(pb::load_model_request::Source::Paths(ModelPaths {
                config_path: format!("{model_dir}/config.json"),
                weights_path: format!("{model_dir}/model.safetensors"),
                tokenizer_path: tokenizer_path.to_string(),
                extra_weights_paths: vec![format!(
                    "/tmp/volvoxai-missing-adapter-{}.safetensors",
                    std::process::id()
                )],
            })),
            exec: None,
            alias: None,
            edit: Some(ModelEditOptions {
                open_weights_writable: false,
                allow_tensor_updates: true,
                allow_graph_patches: false,
            }),
        },
    );
    assert!(failed_reload.is_err());
    let preserved = run_lm_with_route(model_id, inputs, last_token, None);
    assert_eq!(
        preserved.outputs[0].data, base.outputs[0].data,
        "failed reload corrupted the active model/tokenizer state"
    );
    generate_once(model_id).expect("failed reload corrupted tokenizer state");

    let model_flush_path = format!(
        "/tmp/volvoxai-atomic-model-{}.safetensors",
        std::process::id()
    );
    let flushed_model: SaveModelWeightsResponse = call(
        "/volvoxai.v1.VolvoxAiService/SaveModelWeights",
        &SaveModelWeightsRequest {
            model_id: model_id.to_string(),
            weight_paths: vec![model_flush_path.clone()],
            atomic: true,
        },
    )
    .expect("atomic SaveModelWeights");
    assert_eq!(flushed_model.weight_paths, vec![model_flush_path.clone()]);
    assert!(!flushed_model.weight_files.is_empty());
    std::fs::remove_file(&model_flush_path).ok();

    let zero_a = f32_tensor("grpc.h0.qkv.lora_a", vec![64, 1], 0.0);
    let zero_b = f32_tensor("grpc.h0.qkv.lora_b", vec![1, 192], 0.0);
    let mut invalid_manifest = lora_manifest(&zero_a, &zero_b);
    invalid_manifest.targets[0].alpha = 0.0;
    let invalid_alpha = call::<_, AdapterInfo>(
        "/volvoxai.v1.VolvoxAiService/StageAdapter",
        &StageAdapterRequest {
            model_id: model_id.to_string(),
            manifest: Some(invalid_manifest),
            tensors: vec![zero_a.clone(), zero_b.clone()],
            activate: false,
        },
    );
    assert!(invalid_alpha.is_err(), "accepted a zero LoRA alpha");

    let staged: AdapterInfo = call(
        "/volvoxai.v1.VolvoxAiService/StageAdapter",
        &StageAdapterRequest {
            model_id: model_id.to_string(),
            manifest: Some(lora_manifest(&zero_a, &zero_b)),
            tensors: vec![zero_a, zero_b],
            activate: false,
        },
    )
    .expect("StageAdapter");
    let staged_target = &staged
        .manifest
        .as_ref()
        .expect("normalized staged manifest")
        .targets[0];
    assert_eq!(staged_target.alpha, 1.0);
    assert_eq!(staged_target.scale, Some(1.0));
    let staged_ref = staged.adapter.clone().expect("staged adapter ref");

    let listed: ListAdaptersResponse = call(
        "/volvoxai.v1.VolvoxAiService/ListAdapters",
        &ListAdaptersRequest {
            model_id: model_id.to_string(),
            adapter_id: None,
        },
    )
    .expect("ListAdapters");
    assert_eq!(listed.adapters.len(), 1);

    let selection = AdapterSelection {
        routes: vec![AdapterRoute {
            adapter: Some(staged_ref.clone()),
            scale: None,
        }],
    };
    let routed = run_lm_with_route(model_id, inputs, last_token, Some(selection.clone()));
    assert_eq!(
        routed.outputs[0].data, base.outputs[0].data,
        "zero LoRA changed output"
    );

    let _: AdapterState = call(
        "/volvoxai.v1.VolvoxAiService/ActivateAdapter",
        &ActivateAdapterRequest {
            model_id: model_id.to_string(),
            selection: Some(selection),
        },
    )
    .expect("ActivateAdapter");
    let default_routed = run_lm_with_route(model_id, inputs, last_token, None);
    assert_eq!(default_routed.outputs[0].data, base.outputs[0].data);
    let explicit_base = run_lm_with_route(
        model_id,
        inputs,
        last_token,
        Some(AdapterSelection::default()),
    );
    assert_eq!(explicit_base.outputs[0].data, base.outputs[0].data);

    let next_a = f32_tensor("grpc.h0.qkv.lora_a", vec![64, 1], 0.001);
    let next_b = f32_tensor("grpc.h0.qkv.lora_b", vec![1, 192], 0.001);
    let updated: AdapterInfo = call(
        "/volvoxai.v1.VolvoxAiService/UpdateAdapter",
        &UpdateAdapterRequest {
            model_id: model_id.to_string(),
            parent: Some(staged_ref.clone()),
            updates: vec![
                AdapterTensorUpdate {
                    tensor: Some(next_a.clone()),
                    mode: AdapterUpdateMode::AdapterUpdateAssign as i32,
                },
                AdapterTensorUpdate {
                    tensor: Some(next_b.clone()),
                    mode: AdapterUpdateMode::AdapterUpdateAssign as i32,
                },
            ],
            activate: true,
            optimizer_step: 1,
        },
    )
    .expect("UpdateAdapter");
    let updated_ref = updated.adapter.clone().expect("updated adapter ref");
    assert_ne!(updated_ref.version_id, staged_ref.version_id);
    assert_eq!(updated.parent_version_id, staged_ref.version_id);
    let adapted = run_lm_with_route(model_id, inputs, last_token, None);
    assert_ne!(
        adapted.outputs[0].data, base.outputs[0].data,
        "nonzero LoRA had no effect"
    );
    assert!(f32_vec(&adapted.outputs[0].data)
        .iter()
        .all(|x| x.is_finite()));

    let mut fake_batch_inputs = inputs.to_vec();
    for input in &mut fake_batch_inputs {
        input.shape = vec![2, 128];
    }
    let fake_batch = call::<_, RunResponse>(
        "/volvoxai.v1.VolvoxAiService/Run",
        &RunRequest {
            model_id: model_id.to_string(),
            inputs: fake_batch_inputs,
            output_names: vec![],
            last_token,
            exec: None,
            adapters: Some(AdapterSelection {
                routes: vec![
                    AdapterRoute {
                        adapter: Some(staged_ref.clone()),
                        scale: None,
                    },
                    AdapterRoute {
                        adapter: Some(updated_ref.clone()),
                        scale: None,
                    },
                ],
            }),
        },
    );
    assert!(fake_batch.is_err(), "accepted a client-forged batch shape");

    let path = format!("/tmp/volvoxai-grpc-lora-{}.safetensors", std::process::id());
    let saved: SaveAdapterResponse = call(
        "/volvoxai.v1.VolvoxAiService/SaveAdapter",
        &SaveAdapterRequest {
            model_id: model_id.to_string(),
            adapter: Some(updated_ref.clone()),
            path: path.clone(),
            atomic: true,
        },
    )
    .expect("SaveAdapter");
    assert!(string_entry_value(
        &saved
            .weight_file
            .as_ref()
            .expect("saved safetensors info")
            .metadata,
        "volvox_adapter_manifest",
    )
    .is_some());
    let expected_manifest = saved
        .adapter
        .as_ref()
        .and_then(|adapter| adapter.manifest.clone())
        .expect("flushed canonical manifest");
    let loaded: AdapterInfo = call(
        "/volvoxai.v1.VolvoxAiService/LoadAdapter",
        &LoadAdapterRequest {
            model_id: model_id.to_string(),
            path: path.clone(),
            adapter_id: Some("grpc-lora".to_string()),
            manifest: Some(expected_manifest),
            activate: false,
        },
    )
    .expect("LoadAdapter after flush");
    assert_eq!(loaded.optimizer_step, 1);
    assert_eq!(loaded.parent_version_id, staged_ref.version_id);
    assert_eq!(
        loaded
            .manifest
            .as_ref()
            .and_then(|manifest| string_entry_value(&manifest.metadata, "purpose")),
        Some("grpc-roundtrip")
    );
    let loaded_ref = loaded.adapter.clone().expect("loaded adapter ref");
    let loaded_output = run_lm_with_route(
        model_id,
        inputs,
        last_token,
        Some(AdapterSelection {
            routes: vec![AdapterRoute {
                adapter: Some(loaded_ref.clone()),
                scale: None,
            }],
        }),
    );
    assert_eq!(loaded_output.outputs[0].data, adapted.outputs[0].data);

    let _: AdapterState = call(
        "/volvoxai.v1.VolvoxAiService/ActivateAdapter",
        &ActivateAdapterRequest {
            model_id: model_id.to_string(),
            selection: Some(AdapterSelection {
                routes: vec![AdapterRoute {
                    adapter: Some(staged_ref.clone()),
                    scale: None,
                }],
            }),
        },
    )
    .expect("activate pre-merge adapter");

    let merged: AdapterState = call(
        "/volvoxai.v1.VolvoxAiService/MergeAdapter",
        &MergeAdapterRequest {
            model_id: model_id.to_string(),
            adapter: Some(updated_ref.clone()),
        },
    )
    .expect("MergeAdapter");
    assert!(merged.merged.is_some());
    let activate_while_merged = call::<_, AdapterState>(
        "/volvoxai.v1.VolvoxAiService/ActivateAdapter",
        &ActivateAdapterRequest {
            model_id: model_id.to_string(),
            selection: Some(AdapterSelection::default()),
        },
    );
    assert!(activate_while_merged.is_err());
    let merged_output = run_lm_with_route(model_id, inputs, last_token, None);
    let max_merge_diff = f32_vec(&merged_output.outputs[0].data)
        .iter()
        .zip(f32_vec(&adapted.outputs[0].data))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_merge_diff < 1.0e-3,
        "merged and unmerged adapter differ by {max_merge_diff}"
    );
    let remove_saved = call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/RemoveAdapter",
        &RemoveAdapterRequest {
            model_id: model_id.to_string(),
            adapter: Some(staged_ref.clone()),
        },
    );
    assert!(
        remove_saved.is_err(),
        "removed the active version saved for unmerge"
    );
    let _: AdapterState = call(
        "/volvoxai.v1.VolvoxAiService/UnmergeAdapter",
        &UnmergeAdapterRequest {
            model_id: model_id.to_string(),
            adapter: Some(updated_ref.clone()),
        },
    )
    .expect("UnmergeAdapter");
    let restored = run_lm_with_route(model_id, inputs, last_token, None);
    assert_eq!(
        restored.outputs[0].data, base.outputs[0].data,
        "unmerge did not restore base plus saved zero adapter"
    );

    let _: AdapterState = call(
        "/volvoxai.v1.VolvoxAiService/ActivateAdapter",
        &ActivateAdapterRequest {
            model_id: model_id.to_string(),
            selection: Some(AdapterSelection::default()),
        },
    )
    .expect("activate base");
    for reference in [staged_ref, updated_ref, loaded_ref] {
        call::<_, pb::Empty>(
            "/volvoxai.v1.VolvoxAiService/RemoveAdapter",
            &RemoveAdapterRequest {
                model_id: model_id.to_string(),
                adapter: Some(reference),
            },
        )
        .expect("RemoveAdapter");
    }

    let reloaded: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/LoadModel",
        &LoadModelRequest {
            source: Some(pb::load_model_request::Source::Paths(ModelPaths {
                config_path: format!("{model_dir}/config.json"),
                weights_path: format!("{model_dir}/model.safetensors"),
                tokenizer_path: tokenizer_path.to_string(),
                extra_weights_paths: vec![path.clone()],
            })),
            exec: None,
            alias: None,
            edit: Some(ModelEditOptions {
                open_weights_writable: false,
                allow_tensor_updates: true,
                allow_graph_patches: false,
            }),
        },
    )
    .expect("LoadModel with adapter checkpoint in extra_weights_paths");
    let extras: ListAdaptersResponse = call(
        "/volvoxai.v1.VolvoxAiService/ListAdapters",
        &ListAdaptersRequest {
            model_id: reloaded.model_id.clone(),
            adapter_id: None,
        },
    )
    .expect("list auto-staged adapter");
    assert_eq!(extras.adapters.len(), 1);
    let extra_ref = extras.adapters[0]
        .adapter
        .clone()
        .expect("auto-staged adapter ref");
    let extra_output = run_lm_with_route(
        &reloaded.model_id,
        inputs,
        last_token,
        Some(AdapterSelection {
            routes: vec![AdapterRoute {
                adapter: Some(extra_ref),
                scale: None,
            }],
        }),
    );
    assert_eq!(extra_output.outputs[0].data, adapted.outputs[0].data);
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel",
        &ModelRef {
            model_id: reloaded.model_id,
        },
    )
    .expect("unload model with auto-staged adapter");
    std::fs::remove_file(&path).ok();
    println!("[tinystories_1m] adapter lifecycle and persistence passed");
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
        "/volvoxai.v1.VolvoxAiService/LoadModel",
        &LoadModelRequest {
            source: Some(pb::load_model_request::Source::Paths(ModelPaths {
                config_path: format!("{dir}/config.json"),
                weights_path: format!("{dir}/model.safetensors"),
                tokenizer_path: tokenizer_path.clone(),
                extra_weights_paths: vec![],
            })),
            exec: None,
            alias: None,
            edit: Some(ModelEditOptions {
                open_weights_writable: false,
                allow_tensor_updates: true,
                allow_graph_patches: false,
            }),
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

    if name.contains("fp16") || name.contains("int8") {
        let weight_name = cfg["nodes"]
            .as_array()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find_map(|node| node["inputs"]["weight"].as_str())
            })
            .expect("model has no weight input");
        let tensor: Tensor = call(
            "/volvoxai.v1.VolvoxAiService/GetTensor",
            &GetTensorRequest {
                model_id: load.model_id.clone(),
                name: weight_name.to_string(),
            },
        )
        .expect("typed GetTensor");
        if name.contains("fp16") {
            assert_eq!(tensor.dtype, 14);
        } else {
            assert!(tensor.dtype == 5 || tensor.dtype == 6);
        }
        let _: TensorSpec = call(
            "/volvoxai.v1.VolvoxAiService/SetTensor",
            &SetTensorRequest {
                model_id: load.model_id.clone(),
                tensor: Some(tensor),
                persist_safetensors: false,
                safetensors_path: None,
            },
        )
        .expect("typed GetTensor/SetTensor round trip");
    }

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
        let ext = if width == 4 && dtype == 16 {
            "i32"
        } else if dtype == 18 || dtype == 14 {
            "f32"
        } else {
            ""
        };
        let file = format!("{dir}/{iname}.{ext}");
        let data = match std::fs::read(&file) {
            Ok(d) if d.len() == numel * width => d,
            _ => vec![0u8; numel * width],
        };
        if iname == "tokens" && dtype == 16 {
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
            access_flags: 0,
            initializer: None,
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

    if name == "efficientdet_lite0_fp32" {
        let weight_name = cfg["nodes"]
            .as_array()
            .and_then(|nodes| {
                nodes
                    .iter()
                    .find_map(|node| node["inputs"]["weight"].as_str())
            })
            .expect("model has no weight input");
        let before: Tensor = call(
            "/volvoxai.v1.VolvoxAiService/GetTensor",
            &GetTensorRequest {
                model_id: load.model_id.clone(),
                name: weight_name.to_string(),
            },
        )
        .expect("GetTensor before unauthorized Run");
        let mut malicious_input = before.clone();
        assert!(!malicious_input.data.is_empty());
        malicious_input.data[0] ^= 1;
        let rejected = call::<_, RunResponse>(
            "/volvoxai.v1.VolvoxAiService/Run",
            &RunRequest {
                model_id: load.model_id.clone(),
                inputs: vec![malicious_input],
                output_names: vec![],
                last_token: None,
                exec: None,
                adapters: None,
            },
        );
        assert!(
            rejected
                .as_ref()
                .is_err_and(|message| message.contains("not a declared graph input")),
            "Run accepted a model weight as an input: {rejected:?}"
        );
        let after: Tensor = call(
            "/volvoxai.v1.VolvoxAiService/GetTensor",
            &GetTensorRequest {
                model_id: load.model_id.clone(),
                name: weight_name.to_string(),
            },
        )
        .expect("GetTensor after unauthorized Run");
        assert_eq!(
            after.data, before.data,
            "rejected Run mutated a model weight"
        );
    }

    let resp: RunResponse = call(
        "/volvoxai.v1.VolvoxAiService/Run",
        &RunRequest {
            model_id: load.model_id.clone(),
            inputs: inputs.clone(),
            output_names: outputs.clone(),
            last_token: if is_lm { last_token } else { None },
            exec: None,
            adapters: None,
        },
    )
    .unwrap_or_else(|e| panic!("[{name}] Run: {e}"));

    assert!(!resp.outputs.is_empty(), "[{name}] no outputs");
    for t in &resp.outputs {
        assert_eq!(t.dtype, 18, "[{name}] output {} is not F32", t.name);
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
        exercise_adapter_lifecycle(
            &load.model_id,
            &inputs,
            last_token,
            &resp,
            &dir,
            &tokenizer_path,
        );
    }

    if !is_lm {
        call::<_, pb::Empty>(
            "/volvoxai.v1.VolvoxAiService/UnloadModel",
            &ModelRef {
                model_id: load.model_id,
            },
        )
        .unwrap_or_else(|e| panic!("[{name}] UnloadModel: {e}"));
    }
}

// Decode a little-endian F32 tensor payload.
fn f32_vec(b: &[u8]) -> Vec<f32> {
    assert!(b.len() % 4 == 0);
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn i32_vec(b: &[u8]) -> Vec<i32> {
    assert!(b.len() % 4 == 0);
    b.chunks_exact(4)
        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn exercise_weightless_paths_model() {
    let directory = std::env::temp_dir().join(format!(
        "volvoxai-grpc-weightless-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("create weightless model directory");
    let config_path = directory.join("config.json");
    std::fs::write(
        &config_path,
        br#"{
          "inputs":{"x":{"shape":[2],"dtype":"float32"}},
          "nodes":[{
            "opType":"Identity",
            "inputs":{"input":"x"},
            "outputs":{"out":"y"},
            "outputs_shape":{"out":[2]}
          }],
          "outputs":["y"]
        }"#,
    )
    .expect("write weightless model config");

    let loaded: LoadModelResponse = call(
        "/volvoxai.v1.VolvoxAiService/LoadModel",
        &LoadModelRequest {
            source: Some(load_model_request::Source::Paths(ModelPaths {
                config_path: config_path.to_string_lossy().into_owned(),
                weights_path: String::new(),
                tokenizer_path: String::new(),
                extra_weights_paths: Vec::new(),
            })),
            exec: None,
            alias: None,
            edit: None,
        },
    )
    .expect("LoadModel accepts a graph with no weight files");
    let response: RunResponse = call(
        "/volvoxai.v1.VolvoxAiService/Run",
        &RunRequest {
            model_id: loaded.model_id.clone(),
            inputs: vec![f32_values_tensor("x", vec![2], &[1.25, -3.5])],
            output_names: Vec::new(),
            last_token: None,
            exec: None,
            adapters: None,
        },
    )
    .expect("run weightless model");
    assert_eq!(response.outputs.len(), 1);
    assert_eq!(f32_vec(&response.outputs[0].data), vec![1.25, -3.5]);
    call::<_, pb::Empty>(
        "/volvoxai.v1.VolvoxAiService/UnloadModel",
        &ModelRef {
            model_id: loaded.model_id,
        },
    )
    .expect("unload weightless model");
    std::fs::remove_dir_all(directory).expect("remove weightless model directory");
}

#[test]
fn models_work_via_synurang_ffi_grpc_boundary() {
    volvoxai::VolvoxAI_Init(); // register the plugin

    let ping: PingResponse =
        call("/volvoxai.v1.VolvoxAiService/Ping", &pb::Empty {}).expect("Ping");
    println!("ping: {}", ping.version);

    exercise_weightless_paths_model();
    exercise_create_model();
    exercise_multi_loss_gradient_accumulation();
    exercise_from_scratch_training_checkpoint();
    exercise_seq2seq_train_step();
    exercise_backend_rejection_preserves_model();

    // GPU shader lookup must not depend on the host process being launched
    // from the repository root.
    let original_directory = std::env::current_dir().expect("current directory");
    let isolated_directory = std::env::temp_dir().join(format!(
        "volvoxai-grpc-gpu-cwd-{}", std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&isolated_directory);
    std::fs::create_dir_all(&isolated_directory).expect("create isolated GPU cwd");
    std::env::set_current_dir(&isolated_directory).expect("enter isolated GPU cwd");
    exercise_gpu_backend(2, "Vulkan", "VOLVOXAI_REQUIRE_VULKAN");
    exercise_gpu_backend(3, "OpenGL", "VOLVOXAI_REQUIRE_OPENGL");
    #[cfg(target_os = "macos")]
    exercise_gpu_backend(4, "Metal", "VOLVOXAI_REQUIRE_METAL");
    std::env::set_current_dir(original_directory).expect("restore test cwd");
    let _ = std::fs::remove_dir_all(isolated_directory);

    for m in [
        "efficientdet_lite0_fp32",
        "efficientdet_lite0_fp16",
        "efficientdet_lite0_int8",
        "tinystories_1m",
    ] {
        run_model(m);
    }
}
