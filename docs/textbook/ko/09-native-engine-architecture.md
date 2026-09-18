# 9장 — 네이티브 런타임

*자신에게 맞는 배지를 읽으세요: 🌱 **아이디어**(누구나, 코드 없이) · 🔧 **만들기**(코드 조금) · 🔬
**심화**(런타임 개발자). 처음이신가요? 🌱 부분만 따라가세요.*

*목표: 같은 VolvoxAI 패키지가 독립 C 프로그램에서 어떻게 실행되는지 이해합니다. 생성된
proto API를 따라가고, 컴파일이 프로바이더 경로를 어떻게 고정하는지, 내부 컨텍스트와 불변 결과가 왜
동시 네이티브 실행을 안전하게 만드는지 배웁니다.*

> 🌱 **핵심 아이디어.** 네이티브 애플리케이션은 다섯 가지를 합니다: 런타임 생성, 모델 로드, 디바이스용
> 컴파일, 비공개 실행 컨텍스트 생성, 결과 읽기. 로드와 컴파일 작업은 공유할 수 있습니다. 변경 가능한
> 요청 상태는 컨텍스트에 속하고, 완료된 결과는 자기 출력 스냅샷을 소유합니다.

## 9.1 브라우저 밖에서도 같은 패키지

네이티브 런타임은 정규 패키지를 사용합니다:

~~~text
model/
├── graph.json
└── model.safetensors
~~~

Graph 루트에는 정확한 판별자가 있습니다:

~~~json
{
  "format": "volvox-graph/v1",
  "dimensions": {}
}
~~~

Graph 위상, 이름 붙은 입력과 출력, dtype, shape, 연산자 속성, 가중치 서술자는 graph.json에 있습니다.
Safetensors 파일은 텐서 바이트를 담습니다. 네이티브와 JavaScript 프런트엔드는 같은 모델 계약을
컴파일합니다.

고정 릴리스 프로그램은 다음과 같습니다:

~~~text
native/volvoxai-lite
native/volvoxai
~~~

첫 프로그램에는 추론만 있습니다. 두 번째에는 학습 명령과 컴파일된 학습 구현이 추가됩니다. 추론
프로그램에는 학습 구현이나 공개 학습 심볼이 없습니다.

## 9.2 밖에는 생성 ID, 안에는 소유권 트리

공개 C 표면은 `proto/volvoxai.proto`에서 생성됩니다. `volvoxai_ffi.h`는 서비스 진입점을,
`volvoxai_lite.h`는 메시지 코덱을 담습니다. 애플리케이션은 생성된 메시지의 정수 ID를 받으며,
엔진 포인터나 손으로 작성한 수명 주기 API를 받지 않습니다.

생성된 핸들러 뒤에서 네이티브 엔진은 다음 **내부** 소유권 트리를 유지합니다:

~~~text
VxRuntime
  VxModel
    VxCompiledModel
      VxExecutionContext
        VxResult
~~~

- **VxRuntime** 은 프로바이더 런타임과 루트 정책을 소유합니다.
- **VxModel** 은 검증된 패키지 스냅샷 하나를 소유합니다.
- **VxCompiledModel** 은 선택되고 컴파일된 프로바이더 경로를 소유합니다.
- **VxExecutionContext** 는 변경 가능한 입력과 요청/디바이스 상태를 소유합니다.
- **VxResult** 는 한 실행의 불변 선언 출력 스냅샷을 소유합니다.

각 내부 자식은 필요한 부모 상태를 유지합니다. 공개 `Release*` 연산은 멱등적으로 해당 공개
ID만 폐기합니다. 이미 수락된 작업과 후손은 내부 참조를 유지하고, 마지막 참조가 없어질
때 물리적 close/drain이 시작됩니다. 따라서 공개 부모 ID를 해제해도 유지된 자식이나
결과는 무효화되지 않습니다.

생성 호출자는 같은 소유권을 ID로 봅니다:

~~~text
runtime_id → model_id → compiled_model_id → context_id → result_id
~~~

## 9.3 생성, 로드, 컴파일

🔧 생성된 FFI와 lite 코덱을 포함합니다. optional, repeated, oneof 필드가 있는 요청은
인코딩해 `_pb` 진입점으로 보냅니다:

~~~c
#include "volvoxai_ffi.h"
#include "volvoxai_lite.h"

VolvoxaiV1CreateRuntimeRequest request;
VolvoxaiV1RuntimeHandle handle;
uint8_t *encoded = NULL, *response = NULL;
size_t encoded_len = 0;
int32_t response_len = 0;
int64_t runtime_id = 0;

volvoxai_v1_create_runtime_request_init(&request);
/* execution_mode가 없으면 DIRECT. */
volvoxai_v1_create_runtime_request_encode(&request, &encoded, &encoded_len);
response = vx_inference_create_runtime_pb(
    encoded, (int32_t)encoded_len, &response_len);
synurang_lite_default_allocator()->deallocate(
    synurang_lite_default_allocator()->context, encoded);
volvoxai_v1_create_runtime_request_free(&request);

volvoxai_v1_runtime_handle_init(&handle);
if (response && volvoxai_v1_runtime_handle_decode(
        &handle, response, (size_t)response_len) == SYNURANG_LITE_OK &&
    handle.field_report &&
    handle.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK) {
    runtime_id = handle.field_runtime_id;
}
vx_inference_free(response);
volvoxai_v1_runtime_handle_free(&handle);
~~~

`LoadModelRequest`는 `runtime_id`, `graph_path`, 반복 `weight_paths`를 담고, 생성된
`ModelHandle`은 `model_id`를 돌려줍니다. `CompileModelRequest`는 그 ID와 타입 있는
`BackendPolicy`를 담고, `CompiledModelHandle`은 `compiled_model_id`를 돌려줍니다. 빈
정책은 CPU를 선호합니다. `REQUIRE`는 백엔드 하나를, `PREFER`는 순서 있는 후보를 받습니다.
연산자 폴백은 별도의 생성 enum입니다. 선택은 컴파일 시점에 끝나고 실행은 다른 프로바이더에서
다시 시도하지 않습니다.

🔬 모든 생성 응답에는 `OperationReport`가 들어 있습니다. 프로그램은 타입 있는 상태와 구조화된
경로 증거로 분기하고 메시지는 진단용으로 씁니다. FFI가 `NULL`을 반환하는 transport/codec
실패는 별도 채널입니다. 전체 load/compile/release 순서는 `examples/c_api_client_raw.c`에 있습니다.

## 9.4 컨텍스트 실행

`RunRequest`는 간단한 무상태 경로입니다. `compiled_model_id`와 생성된 `Tensor` 메시지의
완전한 목록만 담습니다. 엔진은 유지된 Runtime을 유도하며, 호출자는 경쟁하는 runtime ID를
넘길 수 없습니다.

디코드나 재사용하는 변경 상태에는 생성된 `CreateExecutionContext`를 호출합니다.
`ExecutionContextHandle`이 `context_id`와 선언된 입력 사양을 함께 담으므로 별도의 입력 목록
연산은 없습니다. 컴파일된 모델 하나는 여러 컨텍스트를 만들 수 있고, 각 입력·스크래치·디코드
상태·진행 작업은 서로 별칭을 만들지 않습니다. `Execute`, `ExecutePrefix`, `DecodePrefill`,
`DecodeStep`은 context ID와 명시적 텐서를 받습니다. `ResetDecode`는 decode 상태를
초기화합니다. `SelectAdapter`는 전달한 정확한 게시 revision을 고정하고, revision이 없으면
base model로 돌아갑니다. `RebindAdapter`는 model에 가장 최근 게시된 단일 revision을
채택합니다. Adapter 연산은 지정된 context만 변경합니다.

실행은 Graph 선언과 다른 텐서 이름, dtype, shape, 바이트 수를 거부합니다.

실행은 모든 선언 출력을 정확히 한 번 게시합니다. 변경 가능한 중간 텐서나 빌린 작업 공간 포인터는
노출하지 않습니다.

## 9.5 안정된 결과와 읽기

`Run`과 컨텍스트 실행은 `result_id`와 `execution_id`를 담은 `ExecutionResultHandle`을
반환합니다. 결과는 뒤의 호출과 컨텍스트 해제 후에도 읽을 수 있습니다. `GetResult`는 안정된
선언 출력을 나열합니다. `ReadOutput(result_id, name)`은 정확한 inline 바이트를 반환하고,
선택적 `BufferView into`는 호출자 소유 메모리로의 복사를 요청하며 용량이 작으면 필요 크기를
보고합니다.

생성된 `ReleaseResult`, `ReleaseExecutionContext`, `ReleaseCompiledModel`, `ReleaseModel`,
`ReleaseRuntime`으로 ID를 명시적으로 해제합니다. 별도의 Close RPC는 없습니다. 해제는
멱등적으로 그 공개 ID만 폐기합니다. 이미 수락된 작업과 후손은 필요한 내부 참조를 유지하고,
마지막 참조가 없어질 때 물리적 close/drain이 시작됩니다. 결과 스냅샷은 `ReleaseResult`까지
독립적입니다.

## 9.6 프로바이더는 전역이 아니라 인스턴스

엔진 임베더는 `volvoxai_backend.h`의 `VxBackendProvider` 호스트 조합 SPI를 구현할 수
있습니다. 이것은 애플리케이션 수명 주기 API가 아니며 proto 연산을 추가하지 않습니다. 서술자는
생성된 핸들러 뒤에 명시적 프로바이더 런타임, 컴파일 인스턴스, 컨텍스트 인스턴스를
만듭니다:

~~~c
VxBackendProvider provider = {
    .struct_size = sizeof(VxBackendProvider),
    .abi_version = VX_BACKEND_ABI_VERSION,
    .name = "my-npu",
    .user_data = &driver,
    .shape_domain = VX_BACKEND_SHAPE_DOMAIN_CAPABILITY_INIT,
    .runtime_create = provider_runtime_create,
    .runtime_destroy = provider_runtime_destroy,
    .compile = provider_compile,
    .compiled_destroy = provider_compiled_destroy,
    .context_create = provider_context_create,
    .context_execute = provider_context_execute,
    .context_close = provider_context_close,
    .context_destroy = provider_context_destroy,
    .exact_contract_marker = VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER,
    .exact_contract_extent = sizeof(VxBackendProvider),
};

provider.shape_domain.support = VX_BACKEND_SHAPE_DOMAIN_FULL;
~~~

컨텍스트 실행은 `VxBackendOutputSink`를 통해 모든 선언 출력을 씁니다. sink는 write가 반환되기
전에 복사하므로 프로바이더가 변경 가능한 디바이스/스크래치 메모리를 결과에 빌려줄 수 없습니다.
네이티브 조합 루트가 호스트를 만들 때 서술자를 설치하고, 일반 애플리케이션은 생성된 서비스만
씁니다. 콜백 코드와 `user_data`는 그 프로바이더에서 만든 모든 내부 소유자가 끝날 때까지
유효해야 합니다.

## 9.7 컴파일 내부

🌱 컴파일은 준비 작업입니다: Graph 전체를 살피고, 디바이스 경로를 고르고, 메모리를 예약하고, 반복될
비싼 준비를 한 번만 수행합니다.

🔬 프로바이더는 다음을 할 수 있습니다:

- 연산자, dtype, shape, layout, 양자화 지원 범위 검증
- 안전한 인접 패턴 융합
- 상수 가중치 프리패킹
- GPU 파이프라인이나 디바이스 Graph 컴파일
- 스크래치 버퍼 수명 계획
- 컴파일 보고서에 경로와 디바이스 증거 기록

필수 작업을 지원하지 않으면 컴파일이 실패합니다. 연산자 폴백이 허용되면 컴파일된 경로가 실행 전에
그 경계를 기록합니다. 노드는 조용히 건너뛰지 않습니다.

CPU 실행은 이식 가능한 커널과, CPU 및 OS가 지원할 때 런타임에서 고르는 SIMD 마이크로커널을
사용합니다. GPU 통합은 드라이버 라이브러리를 실행 시점에 찾으므로 실행 파일에 벤더 SDK를 의무적으로
링크하지 않습니다.

## 9.8 메모리 계획과 셰이더

임시 텐서는 대개 수명이 짧고 겹치지 않습니다. 컴파일 시점 수명 계획은 이전 값이 죽은 뒤 같은 arena
영역을 재사용합니다:

~~~text
시간 ───────────────────────────────────────────────▶
입력         [==============]
숨김 A              [========]
숨김 B                       [==========]
출력                                  [==========]

arena 슬롯 0 [ 입력 ][ 숨김 B에 재사용 ]
arena 슬롯 1        [ 숨김 A ][ 출력에 재사용 ]
~~~

이는 Graph 의미를 바꾸지 않고 할당 오버헤드와 최대 메모리를 줄입니다. 컨텍스트가 자기 arena를
소유하므로 동시 요청은 격리됩니다.

수명 계획이 컴파일 시점인 것은 토폴로지를 따르기 때문이고, 영역의 *크기* 는 그렇지 않습니다. 심볼릭
차원은 요청이 바인딩하기 전까지 크기가 없기 때문입니다. 따라서 컨텍스트는 현재 shape 바인딩으로부터
arena 크기를 정하고, 더 큰 합법적 바인딩이 오면 기하급수적으로 키웁니다. 성장은 트랜잭션입니다:
허용된 예산에 못 들어가는 바인딩은 후보 상태를 되돌린 생성
`NATIVE_STATUS_OUT_OF_MEMORY`를 반환하고, 직전
바인딩은 그대로 쓸 수 있게 남습니다. 엔진은 컨텍스트마다 dynamic_arena_capacity_bytes,
dynamic_arena_high_water_bytes, dynamic_arena_grow_count 를 추적합니다
(native/src/runtime/runtime_state.h). 배포 예산은 최악의 경우를 짐작하지 말고 이 high-water 값을
기준으로 잡아야 합니다. 8장 §8.6 브라우저 동작의 네이티브 형태입니다.

GPU 셰이더 소스는 정규 템플릿에서 생성합니다. 생성 출력과 임베드 바이트 배열은 빌드 산출물이지 편집
대상이 아닙니다. 개발 중 VOLVOXAI_SHADER_DIR 로 외부 셰이더를 가리킬 수 있고, 런타임은 실제로 그
오버라이드를 사용할 때 한 번만 로그를 남깁니다.

## 9.9 디코드와 과제 애플리케이션

네이티브 런타임은 텐서 이름에 의미를 붙이지 않습니다. 토큰화, 이미지 디코딩, 프롬프트 형식, 샘플링,
탐지 후처리, 로봇 I/O는 애플리케이션에 속합니다.

생성된 네이티브 API는 모든 projection과 같은 모델 중립 연산인 `ExecutePrefix`,
`DecodePrefill`, `DecodeStep`, `ResetDecode`를 노출합니다. 애플리케이션은 이름·shape·바이트를
갖춘 텐서를 보내고 선언된 결과를 읽습니다. 프로바이더는 컴파일 경로가 필요로 하는 디바이스
상태를 내부 컨텍스트에 둘 수 있습니다. JavaScript 애플리케이션은 같은 연산을
`VxInferenceServiceClient`와 `pb` 메시지로 호출합니다.

선택형 과제 애플리케이션은 이미지 전처리, 분류, 탐지, 모델 중립 디코드 연산을 보여줍니다. 이 앱과
배포되는 `native/volvoxai` CLI는 모두 생성 FFI + lite의 소비자이며 두 번째 엔진 수명주기를 만들지
않습니다. `examples/c_api_client_raw.c`는 과제 정책을 뺀 더 작은 임베딩 패턴을 보여 줍니다.

~~~bash
make -C examples native_task_cli

examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.png --boxes boxes --scores scores --max-det 20
~~~

## 9.10 빌드 프로필

~~~bash
make build_native

./native/volvoxai-lite --help
./native/volvoxai --help
~~~

두 프로그램 모두 모델 중립 run 명령을 제공합니다. full 프로그램만 train을 제공합니다:

~~~bash
./native/volvoxai train models/my_model \
  --input input=batch.f32 \
  --targets targets.i32 \
  --logits logits \
  --trainable classifier.weight \
  --microbatches 10 \
  --accumulation-steps 2 \
  --optimizer adamw \
  --output-weights trained.safetensors
~~~

고정 명령 자체가 생성된 public API 클라이언트입니다. 동일한 full 전용 Training 계약을
구현하며, 각 내부 Trainer 소유자는 그래디언트, 옵티마이저 슬롯, 누적 상태, RNG, 작업
가중치를 비공개로 소유하고 commit은 충돌 검사를 거쳐 Model 리비전을 게시합니다.

백엔드 소스 구성은 빌드 시점 Vulkan, OpenGL, CUDA, Metal 옵션으로 제어합니다. 컴파일되지
않았거나 초기화할 수 없는 프로바이더를 요구하면 백엔드 오류를 반환하며 CPU로 바꾸지 않습니다.

## 9.11 방금 배운 것

> 🌱 **아이디어 정리.** 네이티브 추론은 프로세스 전체 기계 하나가 아닙니다. 공유 런타임/모델/컴파일
> 상태, 비공개 실행 컨텍스트, 불변 결과로 이루어진 소유 객체 트리입니다.

- 네이티브 애플리케이션 API는 `proto/volvoxai.proto`에서 생성된 FFI + lite 메시지입니다.
- graph.json 은 정확한 volvox-graph/v1 판별자를 사용합니다.
- 프로바이더와 연산자 정책은 컴파일 중 해결됩니다.
- 컨텍스트는 변경 가능한 요청과 디바이스 상태를 격리합니다.
- 결과는 안정된 이름 붙은 출력 스냅샷을 소유합니다.
- 외부 디바이스는 VxBackendProvider 인스턴스를 구현합니다.
- full 프로필 학습은 생성된 Trainer ID를 쓰고, 엔진은 내부 상태를 비공개로 유지하며
  `CommitTrainer`로 리비전을 원자적으로 게시합니다.
- 추론 프로필과 full 프로필은 물리적으로 분리됩니다.

**다음:** [9C장 — CUDA 백엔드 →](09c-cuda-backend.md)
