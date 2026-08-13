//! End-to-end lifecycle tests through the generated Synurang FFI dispatcher.

use prost::Message;
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use volvoxai::pb::*;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

extern "C" {
    fn Synurang_Invoke_RuntimeService(
        method: *const c_char,
        data: *const c_char,
        data_len: c_int,
        response_len: *mut c_int,
    ) -> *mut c_char;
    fn Synurang_Free(pointer: *mut c_void);
}

#[derive(Clone, PartialEq, Message)]
struct FfiErrorEnvelope {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(int32, tag = "3")]
    grpc_code: i32,
}

fn invoke<Req: Message, Resp: Message + Default>(
    method: &str,
    request: &Req,
) -> Result<Resp, FfiErrorEnvelope> {
    let request_bytes = request.encode_to_vec();
    let method = CString::new(method).unwrap();
    let mut response_len = 0;
    let pointer = unsafe {
        Synurang_Invoke_RuntimeService(
            method.as_ptr(),
            request_bytes.as_ptr().cast::<c_char>(),
            request_bytes.len() as c_int,
            &mut response_len,
        )
    };
    let length = response_len.unsigned_abs() as usize;
    let response_bytes = if length == 0 {
        Vec::new()
    } else {
        assert!(!pointer.is_null(), "dispatcher returned a null payload");
        unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), length).to_vec() }
    };
    if !pointer.is_null() {
        unsafe { Synurang_Free(pointer.cast::<c_void>()) };
    }
    if response_len < 0 {
        return Err(
            FfiErrorEnvelope::decode(response_bytes.as_slice()).unwrap_or(FfiErrorEnvelope {
                code: 13,
                message: "dispatcher returned a malformed error".to_string(),
                grpc_code: 13,
            }),
        );
    }
    Resp::decode(response_bytes.as_slice()).map_err(|error| FfiErrorEnvelope {
        code: 13,
        message: format!("response decode failed: {error}"),
        grpc_code: 13,
    })
}

fn call<Req: Message, Resp: Message + Default>(method: &str, request: &Req) -> Resp {
    invoke(method, request).unwrap_or_else(|error| {
        panic!(
            "{method} failed with native code {} and Synurang error class {}: {}",
            error.code, error.grpc_code, error.message
        )
    })
}

struct Fixture {
    directory: PathBuf,
    graph_path: PathBuf,
    weight_path: Option<PathBuf>,
}

impl Fixture {
    fn graph(format: Option<&str>) -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "volvoxai-runtime-ffi-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("create fixture directory");
        let graph_path = directory.join("graph.json");
        let format = format
            .map(|value| format!(r#""format":"{value}","#))
            .unwrap_or_default();
        std::fs::write(
            &graph_path,
            format!(
                r#"{{
                  {format}
                  "dimensions":{{}},
                  "inputs":{{"x":{{"shape":[2],"dtype":"float32"}}}},
                  "nodes":[{{
                    "id":"identity",
                    "opType":"Identity",
                    "inputs":{{"input":"x"}},
                    "outputs":{{"out":{{
                      "tensor":"y","dtype":"float32","shape":[2]
                    }}}},
                    "params":{{}}
                  }}],
                  "outputs":["y"]
                }}"#
            ),
        )
        .expect("write fixture graph.json");
        Self {
            directory,
            graph_path,
            weight_path: None,
        }
    }

    fn dynamic_graph() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "volvoxai-runtime-ffi-dynamic-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("create dynamic fixture directory");
        let graph_path = directory.join("graph.json");
        std::fs::write(
            &graph_path,
            r#"{
              "format":"volvox-graph/v1",
              "dimensions":{"N":{"min":2,"max":8,"multiple_of":2}},
              "inputs":{"x":{"shape":["N"],"dtype":"float32"}},
              "nodes":[{
                "id":"identity",
                "opType":"Identity",
                "inputs":{"input":"x"},
                "outputs":{"out":{
                  "tensor":"y","dtype":"float32","shape":["N"]
                }},
                "params":{}
              }],
              "outputs":["y"]
            }"#,
        )
        .expect("write dynamic fixture graph.json");
        Self {
            directory,
            graph_path,
            weight_path: None,
        }
    }

    fn training() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "volvoxai-runtime-ffi-training-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("create training fixture directory");
        let graph_path = directory.join("graph.json");
        std::fs::write(
            &graph_path,
            r#"{
              "format":"volvox-graph/v1",
              "dimensions":{},
              "inputs":{"x":{"shape":[1,2],"dtype":"float32"}},
              "nodes":[{
                "id":"matmul",
                "opType":"MatMul",
                "inputs":{"input":"x","weight":"w"},
                "outputs":{"out":{
                  "tensor":"logits","dtype":"float32","shape":[1,2]
                }},
                "params":{}
              }],
              "outputs":["logits"]
            }"#,
        )
        .expect("write training graph.json");
        let weight_path = directory.join("model.safetensors");
        write_f32_safetensors(&weight_path, "w", &[2, 2], &[0.25, -0.40, 0.15, 0.30]);
        Self {
            directory,
            graph_path,
            weight_path: Some(weight_path),
        }
    }

    fn ptq() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "volvoxai-runtime-ffi-ptq-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("create PTQ fixture directory");
        let graph_path = directory.join("graph.json");
        std::fs::write(
            &graph_path,
            r#"{
              "format":"volvox-graph/v1",
              "dimensions":{},
              "inputs":{"x":{"shape":[2,3],"dtype":"float32"}},
              "nodes":[{
                "id":"linear",
                "opType":"Linear",
                "inputs":{"input":"x","weight":"weight","bias":"bias"},
                "outputs":{"out":{
                  "tensor":"y","dtype":"float32","shape":[2,2]
                }},
                "params":{"weight_layout":"dout_din"}
              }],
              "outputs":["y"]
            }"#,
        )
        .expect("write PTQ source graph.json");
        std::fs::write(
            directory.join("template.graph.json"),
            r#"{
              "format":"volvox-graph/v1",
              "dimensions":{},
              "inputs":{"x":{"shape":[2,3],"dtype":"int8"}},
              "nodes":[{
                "id":"qlinear",
                "opType":"QLinear",
                "inputs":{"input":"x","weight":"weight.i8","bias":"bias.i32"},
                "outputs":{"out":{
                  "tensor":"y","dtype":"int8","shape":[2,2]
                }},
                "params":{}
              }],
              "outputs":["y"]
            }"#,
        )
        .expect("write PTQ template graph");
        let weight_path = directory.join("model.safetensors");
        write_ptq_safetensors(&weight_path);
        Self {
            directory,
            graph_path,
            weight_path: Some(weight_path),
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn path_string(path: &Path) -> String {
    path.to_str().expect("UTF-8 fixture path").to_string()
}

fn write_f32_safetensors(path: &Path, name: &str, shape: &[usize], values: &[f32]) {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    let mut header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":{},"data_offsets":[0,{}]}}}}"#,
        serde_json::to_string(shape).unwrap(),
        values.len() * std::mem::size_of::<f32>(),
    )
    .into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = Vec::with_capacity(8 + header.len() + values.len() * 4);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write safetensors fixture");
}

fn write_ptq_safetensors(path: &Path) {
    let weights = [1.0f32, 2.0, -1.0, -2.0, 0.5, 3.0];
    let bias = [0.25f32, -0.5];
    let weight_bytes = weights.len() * std::mem::size_of::<f32>();
    let total_bytes = weight_bytes + bias.len() * std::mem::size_of::<f32>();
    let mut header = format!(
        r#"{{"weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{weight_bytes}]}},"bias":{{"dtype":"F32","shape":[2],"data_offsets":[{weight_bytes},{total_bytes}]}}}}"#,
    )
    .into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = Vec::with_capacity(8 + header.len() + total_bytes);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    for value in weights.into_iter().chain(bias) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write PTQ safetensors fixture");
}

fn f32_tensor(name: &str, values: [f32; 2]) -> Tensor {
    f32_values_tensor(name, &values)
}

fn f32_values_tensor(name: &str, values: &[f32]) -> Tensor {
    Tensor {
        name: name.to_string(),
        shape: vec![values.len() as i64],
        dtype: DataType::F32 as i32,
        data: values.iter().flat_map(|value| value.to_le_bytes()).collect(),
        location: MemoryLocation::Host as i32,
    }
}

fn f32_training_tensor(values: [f32; 2]) -> Tensor {
    Tensor {
        name: "x".to_string(),
        shape: vec![1, 2],
        dtype: DataType::F32 as i32,
        data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        location: MemoryLocation::Host as i32,
    }
}

fn f32_ptq_tensor(values: [f32; 6]) -> Tensor {
    Tensor {
        name: "x".to_string(),
        shape: vec![2, 3],
        dtype: DataType::F32 as i32,
        data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        location: MemoryLocation::Host as i32,
    }
}

fn decode_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn create_runtime() -> RuntimeHandle {
    call(
        "/volvoxai.runtime.RuntimeService/CreateRuntime",
        &CreateRuntimeRequest {
            debug: false,
            cpu_threads: 1,
        },
    )
}

fn create_context_with_options(
    runtime_id: &str,
    graph_path: &Path,
    decode_row_mode: DecodeRowMode,
    require_incremental: bool,
) -> (String, String, String) {
    let model: ModelHandle = call(
        "/volvoxai.runtime.RuntimeService/LoadModel",
        &LoadModelRequest {
            runtime_id: runtime_id.to_string(),
            graph_path: path_string(graph_path),
            weight_paths: Vec::new(),
        },
    );
    let compiled: CompiledModelHandle = call(
        "/volvoxai.runtime.RuntimeService/CompileModel",
        &CompileModelRequest {
            model_id: model.model_id.clone(),
            policy: Some(BackendPolicy {
                mode: BackendPolicyMode::Require as i32,
                backends: vec!["cpu".to_string()],
                operator_fallback: OperatorFallback::Forbid as i32,
            }),
        },
    );
    let context: ExecutionContextHandle = call(
        "/volvoxai.runtime.RuntimeService/CreateExecutionContext",
        &CreateExecutionContextRequest {
            compiled_model_id: compiled.compiled_model_id.clone(),
            decode_row_mode: decode_row_mode as i32,
            require_incremental,
        },
    );
    assert_eq!(context.inputs.len(), 1);
    assert_eq!(context.inputs[0].name, "x");
    assert_eq!(context.inputs[0].dimensions.len(), 1);
    assert_eq!(context.inputs[0].dimensions[0].symbol, "");
    assert_eq!(context.inputs[0].dimensions[0].min, 2);
    assert_eq!(context.inputs[0].dimensions[0].max, 2);
    assert_eq!(context.inputs[0].dtype, DataType::F32 as i32);
    (
        model.model_id,
        compiled.compiled_model_id,
        context.context_id,
    )
}

fn create_context(runtime_id: &str, graph_path: &Path) -> (String, String, String) {
    create_context_with_options(runtime_id, graph_path, DecodeRowMode::Disabled, false)
}

fn execute(context_id: &str, values: [f32; 2]) -> ExecutionResultHandle {
    call(
        "/volvoxai.runtime.RuntimeService/Execute",
        &ExecuteRequest {
            context_id: context_id.to_string(),
            inputs: vec![f32_tensor("x", values)],
        },
    )
}

fn read_output(result_id: &str) -> Tensor {
    call(
        "/volvoxai.runtime.RuntimeService/ReadOutput",
        &ReadOutputRequest {
            result_id: result_id.to_string(),
            name: "y".to_string(),
        },
    )
}

fn trainer_step_request(
    trainer_id: &str,
    input: Tensor,
    target: i32,
    optimizer: TrainingOptimizerKind,
    accumulation_steps: u32,
) -> TrainStepRequest {
    TrainStepRequest {
        trainer_id: trainer_id.to_string(),
        inputs: vec![input],
        losses: vec![CrossEntropyLoss {
            name: "loss".to_string(),
            logits_name: "logits".to_string(),
            targets: vec![target],
            ignore_index: None,
            row_index: None,
            weight: None,
            normalizer: if accumulation_steps > 1 { 2.0 } else { 0.0 },
        }],
        trainable_names: vec!["w".to_string()],
        optimizer: Some(TrainerOptimizerOptions {
            kind: optimizer as i32,
            learning_rate: Some(0.05),
            beta1: None,
            beta2: None,
            epsilon: None,
            weight_decay: Some(if optimizer == TrainingOptimizerKind::Adamw {
                0.01
            } else {
                0.0
            }),
            max_gradient_norm: None,
        }),
        accumulation_steps: Some(accumulation_steps),
        flush_accumulation: false,
        reset_accumulation: false,
    }
}

fn trainer_step(
    trainer_id: &str,
    target: i32,
    optimizer: TrainingOptimizerKind,
    accumulation_steps: u32,
) -> TrainStepResult {
    call(
        "/volvoxai.runtime.RuntimeService/TrainStep",
        &trainer_step_request(
            trainer_id,
            f32_training_tensor([1.0, -0.5]),
            target,
            optimizer,
            accumulation_steps,
        ),
    )
}

#[test]
fn contexts_are_independent_and_results_survive_parent_release() {
    let fixture = Fixture::graph(Some("volvox-graph/v1"));
    let runtime = create_runtime();
    let (model_id, compiled_id, context_a) =
        create_context(&runtime.runtime_id, &fixture.graph_path);
    let context_b: ExecutionContextHandle = call(
        "/volvoxai.runtime.RuntimeService/CreateExecutionContext",
        &CreateExecutionContextRequest {
            compiled_model_id: compiled_id.clone(),
            decode_row_mode: DecodeRowMode::Disabled as i32,
            require_incremental: false,
        },
    );

    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel",
        &ModelRef { model_id },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseCompiledModel",
        &CompiledModelRef {
            compiled_model_id: compiled_id,
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );

    let context_b_id = context_b.context_id;
    let thread_a = std::thread::spawn({
        let context_a = context_a.clone();
        move || execute(&context_a, [1.25, -3.5])
    });
    let thread_b = std::thread::spawn({
        let context_b_id = context_b_id.clone();
        move || execute(&context_b_id, [7.0, 9.0])
    });
    let result_a = thread_a.join().expect("context A thread");
    let result_b = thread_b.join().expect("context B thread");
    assert_ne!(result_a.execution_id, result_b.execution_id);

    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseExecutionContext",
        &ExecutionContextRef {
            context_id: context_a.clone(),
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseExecutionContext",
        &ExecutionContextRef {
            context_id: context_a.clone(),
        },
    );
    let closed_failure = invoke::<_, ExecutionResultHandle>(
        "/volvoxai.runtime.RuntimeService/Execute",
        &ExecuteRequest {
            context_id: context_a.clone(),
            inputs: vec![f32_tensor("x", [1.0, 2.0])],
        },
    )
    .expect_err("closed context accepted execution");
    assert_eq!(closed_failure.code, -3);
    assert_eq!(closed_failure.grpc_code, 9);
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseExecutionContext",
        &ExecutionContextRef {
            context_id: context_a,
        },
    );

    assert_eq!(
        decode_f32(&read_output(&result_a.result_id)),
        vec![1.25, -3.5]
    );
    assert_eq!(
        decode_f32(&read_output(&result_b.result_id)),
        vec![7.0, 9.0]
    );
    let result_info: ResultInfo = call(
        "/volvoxai.runtime.RuntimeService/GetResult",
        &ResultRef {
            result_id: result_a.result_id.clone(),
        },
    );
    assert_eq!(result_info.outputs.len(), 1);
    assert_eq!(result_info.outputs[0].name, "y");

    for result_id in [result_a.result_id, result_b.result_id] {
        let _: Empty = call(
            "/volvoxai.runtime.RuntimeService/ReleaseResult",
            &ResultRef { result_id },
        );
    }
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseExecutionContext",
        &ExecutionContextRef {
            context_id: context_b_id.clone(),
        },
    );
    let closed_input = invoke::<_, ExecutionResultHandle>(
        "/volvoxai.runtime.RuntimeService/Execute",
        &ExecuteRequest {
            context_id: context_b_id.clone(),
            inputs: vec![f32_tensor("x", [1.0, 2.0])],
        },
    )
    .expect_err("closed context accepted input");
    assert_eq!(closed_input.code, NativeStatus::HandleDisposed as i32);
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseExecutionContext",
        &ExecutionContextRef {
            context_id: context_b_id,
        },
    );
}

#[test]
fn one_context_executes_multiple_dynamic_shapes_transactionally() {
    let fixture = Fixture::dynamic_graph();
    let runtime = create_runtime();
    let model: ModelHandle = call(
        "/volvoxai.runtime.RuntimeService/LoadModel",
        &LoadModelRequest {
            runtime_id: runtime.runtime_id.clone(),
            graph_path: path_string(&fixture.graph_path),
            weight_paths: Vec::new(),
        },
    );
    let compiled: CompiledModelHandle = call(
        "/volvoxai.runtime.RuntimeService/CompileModel",
        &CompileModelRequest {
            model_id: model.model_id.clone(),
            policy: Some(BackendPolicy {
                mode: BackendPolicyMode::Require as i32,
                backends: vec!["cpu".to_string()],
                operator_fallback: OperatorFallback::Forbid as i32,
            }),
        },
    );
    let context: ExecutionContextHandle = call(
        "/volvoxai.runtime.RuntimeService/CreateExecutionContext",
        &CreateExecutionContextRequest {
            compiled_model_id: compiled.compiled_model_id.clone(),
            decode_row_mode: DecodeRowMode::Disabled as i32,
            require_incremental: false,
        },
    );
    assert_eq!(context.inputs.len(), 1);
    assert_eq!(context.inputs[0].dimensions.len(), 1);
    assert_eq!(context.inputs[0].dimensions[0].symbol, "N");
    assert_eq!(context.inputs[0].dimensions[0].min, 2);
    assert_eq!(context.inputs[0].dimensions[0].max, 8);
    assert_eq!(context.inputs[0].dimensions[0].multiple_of, 2);

    let execute_values = |values: &[f32]| -> ExecutionResultHandle {
        call(
            "/volvoxai.runtime.RuntimeService/Execute",
            &ExecuteRequest {
                context_id: context.context_id.clone(),
                inputs: vec![f32_values_tensor("x", values)],
            },
        )
    };
    let small_values = [1.25f32, -3.5];
    let large_values = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let small = execute_values(&small_values);
    let large = execute_values(&large_values);
    let small_again = execute_values(&small_values);

    let invalid = invoke::<_, ExecutionResultHandle>(
        "/volvoxai.runtime.RuntimeService/Execute",
        &ExecuteRequest {
            context_id: context.context_id.clone(),
            inputs: vec![f32_values_tensor("x", &[9.0, 8.0, 7.0])],
        },
    )
    .expect_err("non-multiple dynamic binding was accepted");
    assert_eq!(invalid.code, NativeStatus::InvalidArgument as i32);

    let recovered = execute_values(&small_values);
    for (result, expected) in [
        (&small, small_values.as_slice()),
        (&large, large_values.as_slice()),
        (&small_again, small_values.as_slice()),
        (&recovered, small_values.as_slice()),
    ] {
        let output = read_output(&result.result_id);
        assert_eq!(output.shape, vec![expected.len() as i64]);
        assert_eq!(decode_f32(&output), expected);
    }

    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseExecutionContext",
        &ExecutionContextRef {
            context_id: context.context_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseExecutionContext",
        &ExecutionContextRef {
            context_id: context.context_id,
        },
    );
    for result in [small, large, small_again, recovered] {
        let _: Empty = call(
            "/volvoxai.runtime.RuntimeService/ReleaseResult",
            &ResultRef {
                result_id: result.result_id,
            },
        );
    }
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseCompiledModel",
        &CompiledModelRef {
            compiled_model_id: compiled.compiled_model_id,
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel",
        &ModelRef {
            model_id: model.model_id,
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );
}

#[test]
fn load_model_rejects_a_noncanonical_graph_before_weight_paths() {
    let fixture = Fixture::graph(Some("Volvox-Graph/V1"));
    let runtime = create_runtime();
    let failure = invoke::<_, ModelHandle>(
        "/volvoxai.runtime.RuntimeService/LoadModel",
        &LoadModelRequest {
            runtime_id: runtime.runtime_id.clone(),
            graph_path: path_string(&fixture.graph_path),
            weight_paths: vec!["/definitely/missing/model.safetensors".to_string()],
        },
    )
    .expect_err("noncanonical graph format was accepted");
    assert_eq!(failure.code, -1);
    assert_eq!(failure.grpc_code, 3);
    assert!(failure.message.contains("volvox-graph/v1"));

    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );
}

#[test]
fn ordered_backend_policy_reports_the_selected_fallback() {
    let fixture = Fixture::graph(Some("volvox-graph/v1"));
    let runtime = create_runtime();
    let model: ModelHandle = call(
        "/volvoxai.runtime.RuntimeService/LoadModel",
        &LoadModelRequest {
            runtime_id: runtime.runtime_id.clone(),
            graph_path: path_string(&fixture.graph_path),
            weight_paths: Vec::new(),
        },
    );
    let compiled: CompiledModelHandle = call(
        "/volvoxai.runtime.RuntimeService/CompileModel",
        &CompileModelRequest {
            model_id: model.model_id.clone(),
            policy: Some(BackendPolicy {
                mode: BackendPolicyMode::Prefer as i32,
                backends: vec!["not-registered".to_string(), "cpu".to_string()],
                operator_fallback: OperatorFallback::Forbid as i32,
            }),
        },
    );
    let report = compiled.report.as_ref().expect("compile report");
    assert_eq!(report.backend, "cpu");
    assert_eq!(report.candidate_count, 2);
    assert!(report.tier_fallback_used);
    assert_eq!(
        report.candidate_outcomes,
        "not-registered:unavailable;cpu:selected"
    );

    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseCompiledModel",
        &CompiledModelRef {
            compiled_model_id: compiled.compiled_model_id,
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel",
        &ModelRef {
            model_id: model.model_id,
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );
}

#[test]
fn decode_seed_step_and_reset_use_immutable_results() {
    let fixture = Fixture::graph(Some("volvox-graph/v1"));
    let runtime = create_runtime();
    let (model_id, compiled_id, context_id) = create_context_with_options(
        &runtime.runtime_id,
        &fixture.graph_path,
        DecodeRowMode::Auto,
        true,
    );

    let mut wrong_shape = f32_tensor("x", [2.5, -4.0]);
    wrong_shape.shape = vec![1, 2];
    let rejected_shape = invoke::<_, ExecutionResultHandle>(
        "/volvoxai.runtime.RuntimeService/DecodeSeed",
        &DecodeSeedRequest {
            context_id: context_id.clone(),
            inputs: vec![wrong_shape],
        },
    )
    .expect_err("same-byte input with the wrong declared shape was accepted");
    assert_eq!(rejected_shape.code, NativeStatus::InvalidArgument as i32);
    assert!(rejected_shape.message.contains("rank"));

    let mut device_input = f32_tensor("x", [2.5, -4.0]);
    device_input.location = MemoryLocation::Device as i32;
    let rejected_location = invoke::<_, ExecutionResultHandle>(
        "/volvoxai.runtime.RuntimeService/DecodeSeed",
        &DecodeSeedRequest {
            context_id: context_id.clone(),
            inputs: vec![device_input],
        },
    )
    .expect_err("device input bytes were treated as host memory");
    assert_eq!(rejected_location.code, NativeStatus::InvalidArgument as i32);
    assert!(rejected_location.message.contains("MEMORY_LOCATION_HOST"));

    let seed: ExecutionResultHandle = call(
        "/volvoxai.runtime.RuntimeService/DecodeSeed",
        &DecodeSeedRequest {
            context_id: context_id.clone(),
            inputs: vec![f32_tensor("x", [2.5, -4.0])],
        },
    );
    let seed_report = seed.report.as_ref().expect("decode seed report");
    assert_eq!(seed_report.stage, OperationStage::Decode as i32);
    assert!(
        seed_report.decode_state.contains("seeded=1"),
        "unexpected decode seed report: {seed_report:?}"
    );
    assert_eq!(decode_f32(&read_output(&seed.result_id)), vec![2.5, -4.0]);

    let step: ExecutionResultHandle = call(
        "/volvoxai.runtime.RuntimeService/DecodeStep",
        &DecodeStepRequest {
            context_id: context_id.clone(),
            position: None,
            inputs: Vec::new(),
        },
    );
    assert_ne!(seed.execution_id, step.execution_id);
    assert_eq!(decode_f32(&read_output(&step.result_id)), vec![2.5, -4.0]);

    let reset: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/ResetDecode",
        &ExecutionContextRef {
            context_id: context_id.clone(),
        },
    );
    assert_eq!(reset.stage, OperationStage::Decode as i32);
    assert!(reset.decode_state.contains("seeded=0"));

    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseExecutionContext",
        &ExecutionContextRef {
            context_id: context_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseExecutionContext",
        &ExecutionContextRef { context_id },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseCompiledModel",
        &CompiledModelRef {
            compiled_model_id: compiled_id,
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel",
        &ModelRef { model_id },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );

    assert_eq!(decode_f32(&read_output(&seed.result_id)), vec![2.5, -4.0]);
    assert_eq!(decode_f32(&read_output(&step.result_id)), vec![2.5, -4.0]);
    for result_id in [seed.result_id, step.result_id] {
        let _: Empty = call(
            "/volvoxai.runtime.RuntimeService/ReleaseResult",
            &ResultRef { result_id },
        );
    }
}

#[test]
fn revisions_and_adapter_selection_are_explicit() {
    let fixture = Fixture::graph(Some("volvox-graph/v1"));
    let runtime = create_runtime();
    let (model_id, compiled_id, context_id) =
        create_context(&runtime.runtime_id, &fixture.graph_path);

    let initial: RevisionInfo = call(
        "/volvoxai.runtime.RuntimeService/GetModelRevision",
        &ModelRef {
            model_id: model_id.clone(),
        },
    );
    assert!(initial.graph_id > 0);
    assert!(initial.graph_revision > 0);
    assert!(initial.weight_id > 0);
    assert_eq!(initial.weight_revision, 1);

    let compile_report: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/GetCompiledModelReport",
        &CompiledModelRef {
            compiled_model_id: compiled_id.clone(),
        },
    );
    assert_eq!(compile_report.graph_id, initial.graph_id);
    assert_eq!(compile_report.weight_revision, initial.weight_revision);

    let first: AdapterRevision = call(
        "/volvoxai.runtime.RuntimeService/PublishAdapter",
        &PublishAdapterRequest {
            model_id: model_id.clone(),
            adapter_name: "ffi-route".to_string(),
            package_path: String::new(),
            version_name: "v1".to_string(),
        },
    );
    assert!(first.adapter_id > 0);
    assert_eq!(first.adapter_revision, 1);

    let selected: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/SelectAdapter",
        &SelectAdapterRequest {
            context_id: context_id.clone(),
            adapter_id: first.adapter_id,
            adapter_revision: first.adapter_revision,
        },
    );
    assert_eq!(selected.stage, OperationStage::Adapter as i32);
    assert_eq!(selected.adapter_id, first.adapter_id);
    assert_eq!(selected.adapter_revision, first.adapter_revision);

    let second: AdapterRevision = call(
        "/volvoxai.runtime.RuntimeService/PublishAdapter",
        &PublishAdapterRequest {
            model_id: model_id.clone(),
            adapter_name: "ffi-route".to_string(),
            package_path: String::new(),
            version_name: "v2".to_string(),
        },
    );
    assert_eq!(second.adapter_id, first.adapter_id);
    assert_eq!(second.adapter_revision, first.adapter_revision + 1);

    let current: RevisionInfo = call(
        "/volvoxai.runtime.RuntimeService/GetModelRevision",
        &ModelRef {
            model_id: model_id.clone(),
        },
    );
    assert_eq!(current.adapter_id, second.adapter_id);
    assert_eq!(current.adapter_revision, second.adapter_revision);

    let rebound: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/RebindAdapter",
        &ExecutionContextRef {
            context_id: context_id.clone(),
        },
    );
    assert_eq!(rebound.adapter_id, second.adapter_id);
    assert_eq!(rebound.adapter_revision, second.adapter_revision);

    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseExecutionContext",
        &ExecutionContextRef {
            context_id: context_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseExecutionContext",
        &ExecutionContextRef { context_id },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseCompiledModel",
        &CompiledModelRef {
            compiled_model_id: compiled_id,
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel",
        &ModelRef { model_id },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );
}

#[test]
fn trainer_lifecycle_keeps_steps_private_and_publishes_exact_revisions() {
    let fixture = Fixture::training();
    let weight_path = fixture.weight_path.as_ref().expect("training weights");
    let runtime = create_runtime();
    let model: ModelHandle = call(
        "/volvoxai.runtime.RuntimeService/LoadModel",
        &LoadModelRequest {
            runtime_id: runtime.runtime_id.clone(),
            graph_path: path_string(&fixture.graph_path),
            weight_paths: vec![path_string(weight_path)],
        },
    );
    let initial: RevisionInfo = call(
        "/volvoxai.runtime.RuntimeService/GetModelRevision",
        &ModelRef {
            model_id: model.model_id.clone(),
        },
    );

    let trainer: TrainerHandle = call(
        "/volvoxai.runtime.RuntimeService/CreateTrainer",
        &CreateTrainerRequest {
            model_id: model.model_id.clone(),
            backend: "cpu".to_string(),
            rng_seed: 1234,
        },
    );
    assert_eq!(trainer.inputs.len(), 1);
    assert_eq!(trainer.inputs[0].name, "x");
    assert_eq!(trainer.inputs[0].dimensions.len(), 2);
    assert_eq!(trainer.inputs[0].dimensions[0].min, 1);
    assert_eq!(trainer.inputs[0].dimensions[0].max, 1);
    assert_eq!(trainer.inputs[0].dimensions[1].min, 2);
    assert_eq!(trainer.inputs[0].dimensions[1].max, 2);
    let create_report = trainer.report.as_ref().expect("Trainer create report");
    assert_eq!(create_report.stage, OperationStage::TrainerCreate as i32);
    assert_eq!(create_report.backend, "cpu");
    assert_eq!(create_report.weight_revision, initial.weight_revision);

    let mut wrong_shape = f32_training_tensor([1.0, -0.5]);
    wrong_shape.shape = vec![2];
    let rejected_shape = invoke::<_, TrainStepResult>(
        "/volvoxai.runtime.RuntimeService/TrainStep",
        &trainer_step_request(
            &trainer.trainer_id,
            wrong_shape,
            0,
            TrainingOptimizerKind::Sgd,
            1,
        ),
    )
    .expect_err("same-byte Trainer input with the wrong shape was accepted");
    assert_eq!(rejected_shape.code, NativeStatus::InvalidArgument as i32);

    let mut device_input = f32_training_tensor([1.0, -0.5]);
    device_input.location = MemoryLocation::Device as i32;
    let rejected_location = invoke::<_, TrainStepResult>(
        "/volvoxai.runtime.RuntimeService/TrainStep",
        &trainer_step_request(
            &trainer.trainer_id,
            device_input,
            0,
            TrainingOptimizerKind::Sgd,
            1,
        ),
    )
    .expect_err("device Trainer input bytes were treated as host memory");
    assert_eq!(rejected_location.code, NativeStatus::InvalidArgument as i32);

    let first = trainer_step(&trainer.trainer_id, 0, TrainingOptimizerKind::Sgd, 2);
    assert!(!first.update_applied);
    assert_eq!(first.accumulated_microbatches, 1);
    assert_eq!(first.optimizer_step, 0);
    assert_eq!(first.metrics.len(), 1);
    assert_eq!(first.metrics[0].name, "loss");
    assert_eq!(
        first.report.as_ref().expect("step report").stage,
        OperationStage::TrainerStep as i32
    );

    let pending = invoke::<_, RevisionInfo>(
        "/volvoxai.runtime.RuntimeService/CommitTrainer",
        &TrainerRef {
            trainer_id: trainer.trainer_id.clone(),
        },
    )
    .expect_err("unfinished accumulation was committed");
    assert_eq!(pending.code, -1);
    assert_eq!(pending.grpc_code, 3);
    assert!(pending.message.contains("ACCUMULATION_PENDING"));

    let second = trainer_step(&trainer.trainer_id, 0, TrainingOptimizerKind::Sgd, 2);
    assert!(second.update_applied);
    assert_eq!(second.optimizer_step, 1);
    let before_commit: RevisionInfo = call(
        "/volvoxai.runtime.RuntimeService/GetModelRevision",
        &ModelRef {
            model_id: model.model_id.clone(),
        },
    );
    assert_eq!(before_commit.weight_revision, initial.weight_revision);

    let published: RevisionInfo = call(
        "/volvoxai.runtime.RuntimeService/CommitTrainer",
        &TrainerRef {
            trainer_id: trainer.trainer_id.clone(),
        },
    );
    assert_eq!(published.weight_id, initial.weight_id);
    assert_eq!(published.weight_revision, initial.weight_revision + 1);
    assert_eq!(
        published.report.as_ref().expect("commit report").stage,
        OperationStage::TrainerCommit as i32
    );

    let baseline_path = fixture.directory.join("baseline.safetensors");
    let dirty_path = fixture.directory.join("dirty.safetensors");
    let restored_path = fixture.directory.join("restored.safetensors");
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/ExportTrainerWeights",
        &ExportTrainerWeightsRequest {
            trainer_id: trainer.trainer_id.clone(),
            output_paths: vec![path_string(&baseline_path)],
        },
    );
    let dirty = trainer_step(&trainer.trainer_id, 1, TrainingOptimizerKind::Adamw, 1);
    assert!(dirty.update_applied);
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/ExportTrainerWeights",
        &ExportTrainerWeightsRequest {
            trainer_id: trainer.trainer_id.clone(),
            output_paths: vec![path_string(&dirty_path)],
        },
    );
    let rollback: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/RollbackTrainer",
        &TrainerRef {
            trainer_id: trainer.trainer_id.clone(),
        },
    );
    assert_eq!(rollback.stage, OperationStage::TrainerRollback as i32);
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/ExportTrainerWeights",
        &ExportTrainerWeightsRequest {
            trainer_id: trainer.trainer_id.clone(),
            output_paths: vec![path_string(&restored_path)],
        },
    );
    let baseline_bytes = std::fs::read(&baseline_path).unwrap();
    let dirty_bytes = std::fs::read(&dirty_path).unwrap();
    let restored_bytes = std::fs::read(&restored_path).unwrap();
    assert_ne!(dirty_bytes, baseline_bytes);
    assert_eq!(restored_bytes, baseline_bytes);

    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseTrainer",
        &TrainerRef {
            trainer_id: trainer.trainer_id.clone(),
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseTrainer",
        &TrainerRef {
            trainer_id: trainer.trainer_id.clone(),
        },
    );
    let closed = invoke::<_, TrainStepResult>(
        "/volvoxai.runtime.RuntimeService/TrainStep",
        &trainer_step_request(
            &trainer.trainer_id,
            f32_training_tensor([1.0, -0.5]),
            0,
            TrainingOptimizerKind::Sgd,
            1,
        ),
    )
    .expect_err("closed Trainer accepted input");
    assert_eq!(closed.code, -3);
    assert_eq!(closed.grpc_code, 9);
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseTrainer",
        &TrainerRef {
            trainer_id: trainer.trainer_id,
        },
    );

    let left: TrainerHandle = call(
        "/volvoxai.runtime.RuntimeService/CreateTrainer",
        &CreateTrainerRequest {
            model_id: model.model_id.clone(),
            backend: "cpu".to_string(),
            rng_seed: 1,
        },
    );
    let right: TrainerHandle = call(
        "/volvoxai.runtime.RuntimeService/CreateTrainer",
        &CreateTrainerRequest {
            model_id: model.model_id.clone(),
            backend: "cpu".to_string(),
            rng_seed: 2,
        },
    );
    let left_id = left.trainer_id.clone();
    let right_id = right.trainer_id.clone();
    let left_step =
        std::thread::spawn(move || trainer_step(&left_id, 0, TrainingOptimizerKind::Adamw, 1));
    let right_step =
        std::thread::spawn(move || trainer_step(&right_id, 1, TrainingOptimizerKind::Adamw, 1));
    assert!(
        left_step
            .join()
            .expect("left Trainer thread")
            .update_applied
    );
    assert!(
        right_step
            .join()
            .expect("right Trainer thread")
            .update_applied
    );

    let winner: RevisionInfo = call(
        "/volvoxai.runtime.RuntimeService/CommitTrainer",
        &TrainerRef {
            trainer_id: left.trainer_id.clone(),
        },
    );
    assert_eq!(winner.weight_revision, published.weight_revision + 1);
    let conflict = invoke::<_, RevisionInfo>(
        "/volvoxai.runtime.RuntimeService/CommitTrainer",
        &TrainerRef {
            trainer_id: right.trainer_id.clone(),
        },
    )
    .expect_err("stale Trainer published a second successor");
    assert_eq!(conflict.code, -17);
    assert_eq!(conflict.grpc_code, 10);
    assert!(conflict.message.contains("REVISION_CONFLICT"));

    for trainer_id in [left.trainer_id, right.trainer_id] {
        let _: OperationReport = call(
            "/volvoxai.runtime.RuntimeService/CloseTrainer",
            &TrainerRef {
                trainer_id: trainer_id.clone(),
            },
        );
        let _: Empty = call(
            "/volvoxai.runtime.RuntimeService/ReleaseTrainer",
            &TrainerRef { trainer_id },
        );
    }
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel",
        &ModelRef {
            model_id: model.model_id,
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );
}

#[test]
fn ptq_plan_is_proto_owned_snapshot_scoped_and_revision_checked() {
    let fixture = Fixture::ptq();
    let template_graph_path = fixture.directory.join("template.graph.json");
    let output_directory = fixture.directory.join("quantized");
    let output_graph_path = output_directory.join("graph.json");
    let output_weights_path = output_directory.join("model.safetensors");
    std::fs::create_dir(&output_directory).expect("create PTQ output directory");
    let runtime = create_runtime();
    let model: ModelHandle = call(
        "/volvoxai.runtime.RuntimeService/LoadModel",
        &LoadModelRequest {
            runtime_id: runtime.runtime_id.clone(),
            graph_path: path_string(&fixture.graph_path),
            weight_paths: vec![path_string(
                fixture.weight_path.as_ref().expect("PTQ fixture weights"),
            )],
        },
    );
    let initial: RevisionInfo = call(
        "/volvoxai.runtime.RuntimeService/GetModelRevision",
        &ModelRef {
            model_id: model.model_id.clone(),
        },
    );
    let plan: PtqPlanHandle = call(
        "/volvoxai.runtime.RuntimeService/CreatePtqPlan",
        &CreatePtqPlanRequest {
            model_id: model.model_id.clone(),
            template_graph_path: path_string(&template_graph_path),
            profile_names: vec!["batch-2".to_string()],
            observers: vec![
                PtqObserverSpec {
                    tensor_name: "x".to_string(),
                    dtype: DataType::I8 as i32,
                    scheme: PtqScheme::Symmetric as i32,
                },
                PtqObserverSpec {
                    tensor_name: "y".to_string(),
                    dtype: DataType::I8 as i32,
                    scheme: PtqScheme::Symmetric as i32,
                },
            ],
            layers: vec![PtqLayerSpec {
                mode: PtqMode::W8a8 as i32,
                kind: PtqLayerKind::Qlinear as i32,
                node_index: 0,
                weight_axis: 0,
                input_tensor_name: "x".to_string(),
                output_tensor_name: "y".to_string(),
                source_weight_name: "weight".to_string(),
                packed_weight_name: "weight.i8".to_string(),
                source_bias_name: "bias".to_string(),
                packed_bias_name: "bias.i32".to_string(),
            }],
        },
    );
    assert_eq!(plan.inputs.len(), 1);
    assert_eq!(plan.inputs[0].name, "x");
    assert_eq!(plan.inputs[0].dimensions.len(), 2);
    assert_eq!(plan.inputs[0].dimensions[0].min, 2);
    assert_eq!(plan.inputs[0].dimensions[0].max, 2);
    assert_eq!(plan.inputs[0].dimensions[1].min, 3);
    assert_eq!(plan.inputs[0].dimensions[1].max, 3);
    let create_report = plan.report.as_ref().expect("PTQ create report");
    assert_eq!(create_report.stage, OperationStage::PtqCreate as i32);
    assert_eq!(create_report.backend, "cpu");
    assert!(create_report.route_attested);
    assert_eq!(create_report.weight_revision, initial.weight_revision);

    let before_calibration: PtqPlanInfo = call(
        "/volvoxai.runtime.RuntimeService/InspectPtqPlan",
        &PtqPlanRef {
            ptq_plan_id: plan.ptq_plan_id.clone(),
        },
    );
    assert_eq!(before_calibration.calibration_batches, 0);
    assert_eq!(before_calibration.calibration_samples, 0);
    let before_coverage = before_calibration
        .coverage
        .as_ref()
        .expect("PTQ profile coverage");
    assert!(!before_coverage.complete);
    assert_eq!(before_coverage.profiles.len(), 1);
    assert_eq!(before_coverage.profiles[0].name, "batch-2");
    assert_eq!(before_calibration.tensors.len(), 2);
    assert!(before_calibration
        .tensors
        .iter()
        .all(|parameters| parameters.observed_values == 0 && parameters.scale == 0.0));

    // Native creation copied a private template snapshot. Mutating the caller
    // file now cannot alter the package emitted later.
    std::fs::write(&template_graph_path, r#"{"format":"mutated"}"#)
        .expect("mutate caller template");
    let sample_values = [1.0, -1.0, 0.5, 0.0, 2.0, -1.0];
    let mut wrong_shape = f32_ptq_tensor(sample_values);
    wrong_shape.shape = vec![6];
    let rejected = invoke::<_, PtqCalibrationInfo>(
        "/volvoxai.runtime.RuntimeService/CalibratePtqPlan",
        &CalibratePtqPlanRequest {
            ptq_plan_id: plan.ptq_plan_id.clone(),
            profile_name: "batch-2".to_string(),
            sample_name: "bad-shape".to_string(),
            sample_count: 2,
            inputs: vec![wrong_shape],
        },
    )
    .expect_err("same-byte PTQ input with the wrong shape was accepted");
    assert_eq!(rejected.code, NativeStatus::InvalidArgument as i32);
    assert_eq!(rejected.grpc_code, 3);

    let calibrated: PtqCalibrationInfo = call(
        "/volvoxai.runtime.RuntimeService/CalibratePtqPlan",
        &CalibratePtqPlanRequest {
            ptq_plan_id: plan.ptq_plan_id.clone(),
            profile_name: "batch-2".to_string(),
            sample_name: "sample-0".to_string(),
            sample_count: 2,
            inputs: vec![f32_ptq_tensor(sample_values)],
        },
    );
    assert_eq!(calibrated.calibration_batches, 1);
    assert_eq!(calibrated.calibration_samples, 2);
    assert!(calibrated.coverage_complete);
    assert_eq!(
        calibrated
            .report
            .as_ref()
            .expect("calibration report")
            .stage,
        OperationStage::PtqCalibrate as i32
    );
    let duplicate = invoke::<_, PtqCalibrationInfo>(
        "/volvoxai.runtime.RuntimeService/CalibratePtqPlan",
        &CalibratePtqPlanRequest {
            ptq_plan_id: plan.ptq_plan_id.clone(),
            profile_name: "batch-2".to_string(),
            sample_name: "sample-0".to_string(),
            sample_count: 2,
            inputs: vec![f32_ptq_tensor(sample_values)],
        },
    )
    .expect_err("duplicate calibration sample name was accepted");
    assert_eq!(duplicate.code, NativeStatus::InvalidArgument as i32);

    let inspected: PtqPlanInfo = call(
        "/volvoxai.runtime.RuntimeService/InspectPtqPlan",
        &PtqPlanRef {
            ptq_plan_id: plan.ptq_plan_id.clone(),
        },
    );
    assert_eq!(inspected.calibration_batches, 1);
    assert_eq!(inspected.calibration_samples, 2);
    let coverage = inspected.coverage.as_ref().expect("PTQ coverage");
    assert_eq!(coverage.format, "volvox.ptq-coverage/v1");
    assert!(coverage.complete);
    assert_eq!(coverage.total_batches, 1);
    assert_eq!(coverage.total_samples, 2);
    assert_eq!(coverage.profiles.len(), 1);
    assert_eq!(coverage.profiles[0].name, "batch-2");
    assert_eq!(coverage.profiles[0].batches, 1);
    assert_eq!(coverage.profiles[0].samples, 2);
    let input_parameters = inspected
        .tensors
        .iter()
        .find(|parameters| parameters.tensor_name == "x")
        .expect("input PTQ parameters");
    assert_eq!(input_parameters.observed_values, 6);
    assert_eq!(input_parameters.observed_min, -1.0);
    assert_eq!(input_parameters.observed_max, 2.0);
    assert_eq!(
        inspected
            .revision
            .as_ref()
            .expect("PTQ exact revision")
            .weight_revision,
        initial.weight_revision
    );

    let package: PtqPackageInfo = call(
        "/volvoxai.runtime.RuntimeService/WritePtqPackage",
        &WritePtqPackageRequest {
            ptq_plan_id: plan.ptq_plan_id.clone(),
            output_graph_path: path_string(&output_graph_path),
            output_weights_path: path_string(&output_weights_path),
        },
    );
    assert_eq!(package.graph_path, path_string(&output_graph_path));
    assert_eq!(package.safetensors_path, path_string(&output_weights_path));
    assert_eq!(
        package.report.as_ref().expect("PTQ write report").stage,
        OperationStage::PtqWrite as i32
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(&output_graph_path).expect("read quantized graph.json")
        )
        .expect("parse quantized graph.json")["format"],
        "volvox-graph/v1"
    );
    assert!(output_weights_path.is_file());

    let _: AdapterRevision = call(
        "/volvoxai.runtime.RuntimeService/PublishAdapter",
        &PublishAdapterRequest {
            model_id: model.model_id.clone(),
            adapter_name: "ptq-stale".to_string(),
            package_path: String::new(),
            version_name: String::new(),
        },
    );
    let stale = invoke::<_, PtqPlanInfo>(
        "/volvoxai.runtime.RuntimeService/InspectPtqPlan",
        &PtqPlanRef {
            ptq_plan_id: plan.ptq_plan_id.clone(),
        },
    )
    .expect_err("stale PTQ plan remained inspectable");
    assert_eq!(stale.code, NativeStatus::RevisionConflict as i32);
    assert_eq!(stale.grpc_code, 10);

    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/ClosePtqPlan",
        &PtqPlanRef {
            ptq_plan_id: plan.ptq_plan_id.clone(),
        },
    );
    let closed_calibration = invoke::<_, PtqCalibrationInfo>(
        "/volvoxai.runtime.RuntimeService/CalibratePtqPlan",
        &CalibratePtqPlanRequest {
            ptq_plan_id: plan.ptq_plan_id.clone(),
            profile_name: "batch-2".to_string(),
            sample_name: "closed".to_string(),
            sample_count: 2,
            inputs: vec![f32_ptq_tensor(sample_values)],
        },
    )
    .expect_err("closed PTQ plan accepted calibration");
    assert_eq!(closed_calibration.code, NativeStatus::HandleDisposed as i32);
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleasePtqPlan",
        &PtqPlanRef {
            ptq_plan_id: plan.ptq_plan_id,
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel",
        &ModelRef {
            model_id: model.model_id,
        },
    );
    let _: OperationReport = call(
        "/volvoxai.runtime.RuntimeService/CloseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id.clone(),
        },
    );
    let _: Empty = call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime",
        &RuntimeRef {
            runtime_id: runtime.runtime_id,
        },
    );
}
