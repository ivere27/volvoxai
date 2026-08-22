# 9장 — 네이티브 런타임

*자신에게 맞는 배지를 읽으세요: 🌱 **아이디어**(누구나, 코드 없이) · 🔧 **만들기**(코드 조금) · 🔬
**심화**(런타임 개발자). 처음이신가요? 🌱 부분만 따라가세요.*

*목표: 같은 VolvoxAI 패키지가 독립 C 프로그램에서 어떻게 실행되는지 이해합니다. 공개 불투명 핸들의
수명 주기를 따라가고, 컴파일이 프로바이더 경로를 어떻게 고정하는지, 컨텍스트와 불변 결과가 왜 동시
네이티브 실행을 안전하게 만드는지 배웁니다.*

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
native/volvoxai
native/volvoxai-full
~~~

첫 프로그램에는 추론만 있습니다. 두 번째에는 학습 명령과 컴파일된 학습 구현이 추가됩니다. 추론
프로그램에는 학습 구현이나 공개 학습 심볼이 없습니다.

## 9.2 소유권 트리

공개 헤더 native/include/volvoxai.h 는 다섯 불투명 핸들 형식을 노출합니다:

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

모든 핸들에는 retain/release 연산이 있습니다. 자식은 필요한 부모 상태를 유지하므로 Runtime 변수를
해제해도 이미 유지된 컨텍스트나 결과는 무효가 되지 않습니다. 논리적 close는 새 작업을 거부하며
여러 번 호출해도 안전합니다.

이는 JavaScript 수명 주기의 네이티브 형태입니다:

~~~text
Runtime → Model → CompiledModel → ExecutionContext → ExecutionResult
~~~

## 9.3 생성, 로드, 컴파일

🔧 volvoxai.h 를 포함하고 각 옵션/보고서 구조체를 대응하는 매크로로 초기화합니다:

~~~c
#include "volvoxai.h"
#include <stdio.h>

VxReport report = VX_REPORT_INIT;
VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
VxRuntime* runtime = NULL;

VxStatus status = vx_runtime_create(
    &runtime_options, &runtime, &report);
if (status != VX_STATUS_OK) {
    fprintf(stderr, "%s\n", report.message);
    return 1;
}

const char* weights[] = {
    "model/model.safetensors",
};
VxModelSource source = VX_MODEL_SOURCE_INIT;
source.graph_path = "model/graph.json";
source.weight_paths = weights;
source.weight_path_count = 1;

VxModel* model = NULL;
status = vx_runtime_load_model(runtime, &source, &model, &report);
~~~

로드 과정은 디바이스를 할당하기 전에 Graph와 가중치 서술자를 검증합니다. 컴파일에는 명시적인
프로바이더 정책을 적용합니다:

~~~c
VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
policy.mode = VX_BACKEND_REQUIRE;
policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
const char* required_backends[] = { "cpu" };
policy.backends = required_backends;
policy.backend_count = 1;

VxCompiledModel* compiled = NULL;
status = vx_model_compile(model, &policy, &compiled, &report);
~~~

VX_BACKEND_REQUIRE 는 backends 항목을 정확히 하나 요구합니다. VX_BACKEND_PREFER 는 backends 에
나열된 순서대로 시도하며, 목록이 NULL 이고 개수가 0이면 CPU 하나만 사용하는 기본 선호 정책을
적용합니다. 연산자 폴백은 별도 선택입니다. 선택은 컴파일 시점에 끝납니다. 실행 실패는 보고되며 다른
프로바이더에서 다시 시도하지 않습니다.

🔬 VxReport 는 수명 주기 단계, 상태, 선택한 백엔드, 해당 프로바이더가 보고한 디바이스 식별자(있는
경우), 사유, 메시지, 실행 식별자를 기록합니다. 프로그램은 VxStatus 와 구조화된 보고서 필드로
분기하고, 메시지는 진단 텍스트로 사용합니다.

## 9.4 컨텍스트 실행

컴파일된 모델 하나에서 여러 컨텍스트를 만들 수 있습니다. 입력, 스크래치 저장소, 디코드 상태,
진행 중 작업은 서로 별칭을 만들지 않습니다.

~~~c
VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
VxExecutionContext* context = NULL;
VxTensorBinding input = VX_TENSOR_BINDING_INIT;

status = vx_compiled_model_create_context(
    compiled, &context_options, &context, &report);

input.name = "input";
input.dtype = VX_DTYPE_F32;
input.rank = 2;
input.shape[0] = batch;
input.shape[1] = sequence;
input.data = input_values;
input.byte_size = input_bytes;
input.location = VX_MEMORY_HOST;

VxResult* result = NULL;
if (status == VX_STATUS_OK) {
    status = vx_execution_context_execute(
        context, &input, 1, &result, &report);
}
~~~

vx_execution_context_input_count 와 vx_execution_context_input_spec 으로 선언된 입력을 살핍니다.
실행은 Graph 선언과 다른 바인딩 이름, dtype, shape, 바이트 수를 거부합니다.

실행은 모든 선언 출력을 정확히 한 번 게시합니다. 변경 가능한 중간 텐서나 빌린 작업 공간 포인터는
노출하지 않습니다.

## 9.5 안정된 결과와 읽기

VxResult 는 이후 실행과 컨텍스트 close 뒤에도 읽을 수 있습니다. 필요한 크기를 질의하고, 호출자 소유
저장소를 할당한 뒤 정확한 출력 이름으로 복사합니다:

~~~c
#include <stdlib.h>

size_t required = 0;
status = vx_result_read(
    result, "logits", NULL, 0, &required, &report);

void* output = malloc(required);
if (output != NULL) {
    status = vx_result_read(
        result, "logits", output, required, NULL, &report);
}
~~~

vx_result_output_count 와 vx_result_output_info 는 이름, shape, dtype, 바이트 크기, 메모리 위치를
노출합니다. vx_result_execution_id 는 결과를 만든 실행을 식별합니다.

소유권이 끝나면 명시적으로 해제합니다:

~~~c
vx_execution_context_close(context, &report);
vx_execution_context_release(context);
vx_compiled_model_release(compiled);
vx_model_release(model);
vx_runtime_release(runtime);

/* 스냅샷은 위 핸들과 독립적입니다. */
vx_result_release(result);
free(output);
~~~

## 9.6 프로바이더는 전역이 아니라 인스턴스

외부 디바이스 통합은 volvoxai_backend.h 의 VxBackendProvider 를 구현합니다. 서술자는 명시적인
프로바이더 런타임, 컴파일 인스턴스, 컨텍스트 인스턴스를 만듭니다:

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

VxStatus registration =
    vx_runtime_register_provider(runtime, &provider, &report);
~~~

컨텍스트 실행은 VxBackendOutputSink 를 통해 모든 선언 출력을 씁니다. sink는 write가 반환되기 전에
복사하므로 프로바이더가 변경 가능한 디바이스/스크래치 메모리를 결과에 빌려줄 수 없습니다. 프로바이더
이름과 서술자는 등록 때 복사되지만, 콜백 코드와 user_data 는 그 프로바이더에서 만든 모든 핸들이
끝날 때까지 유효해야 합니다.

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
허용된 예산에 못 들어가는 바인딩은 후보 상태를 되돌린 VX_STATUS_OUT_OF_MEMORY 를 반환하고, 직전
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

공개 네이티브 실행 API에는 별도 prefix/row 진입점이 없습니다. 애플리케이션은 이름 붙은 입력을 묶고,
컨텍스트를 실행하고, 선언된 결과를 읽습니다. 컴파일된 경로가 요구한다면 프로바이더는 컨텍스트 안에
디바이스 상태를 유지할 수 있습니다. JavaScript에서 대응하는 모델 중립 디코드 표면은
ExecutionContext.decode.seed(), step(), reset() 입니다.

선택형 과제 애플리케이션은 이미지 전처리, 분류, 탐지, 모델 중립 디코드 연산을 보여줍니다:

~~~bash
make -C examples native_task_cli

examples/target/bin/volvoxai-tasks detect models/efficientdet_lite0_int8 \
  --image input0=photo.png --boxes boxes --scores scores --max-det 20
~~~

## 9.10 빌드 프로필

~~~bash
make build_native

./native/volvoxai --help
./native/volvoxai-full --help
~~~

두 프로그램 모두 모델 중립 run 명령을 제공합니다. full 프로그램만 train을 제공합니다:

~~~bash
./native/volvoxai-full train models/my_model \
  --input input=batch.f32 \
  --targets targets.i32 \
  --logits logits \
  --trainable classifier.weight \
  --microbatches 10 \
  --accumulation-steps 2 \
  --optimizer adamw \
  --output-weights trained.safetensors
~~~

이 명령은 full 전용 불투명 VxTrainer 수명 주기를 사용합니다. 각 Trainer는 입력, 그래디언트,
옵티마이저 슬롯, 누적 상태, RNG, 작업 가중치를 비공개로 소유합니다. 명시적인 충돌 검사 commit만
Model 리비전을 게시합니다.

백엔드 소스 구성은 빌드 시점 Vulkan, OpenGL, CUDA, Metal 옵션으로 제어합니다. 컴파일되지
않았거나 초기화할 수 없는 프로바이더를 요구하면 백엔드 오류를 반환하며 CPU로 바꾸지 않습니다.

## 9.11 방금 배운 것

> 🌱 **아이디어 정리.** 네이티브 추론은 프로세스 전체 기계 하나가 아닙니다. 공유 런타임/모델/컴파일
> 상태, 비공개 실행 컨텍스트, 불변 결과로 이루어진 소유 객체 트리입니다.

- 네이티브 API는 불투명 vx_* 수명 주기입니다.
- graph.json 은 정확한 volvox-graph/v1 판별자를 사용합니다.
- 프로바이더와 연산자 정책은 컴파일 중 해결됩니다.
- 컨텍스트는 변경 가능한 요청과 디바이스 상태를 격리합니다.
- 결과는 안정된 이름 붙은 출력 스냅샷을 소유합니다.
- 외부 디바이스는 VxBackendProvider 인스턴스를 구현합니다.
- full 프로필 학습은 비공개 불투명 VxTrainer와 원자적 리비전 게시를 사용합니다.
- 추론 프로필과 full 프로필은 물리적으로 분리됩니다.

**다음:** [9C장 — CUDA 백엔드 →](09c-cuda-backend.md)
