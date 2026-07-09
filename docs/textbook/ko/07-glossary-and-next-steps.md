# 7장 — 용어집과 다음 단계

*목표: 모든 용어를 한곳에, 이 저장소를 지나는 구체적 경로, 그리고 읽기를 실력으로 바꾸는 실습.*

> 용어집 표제어는 영어 용어를 그대로 둡니다(논문·코드에서 이 영어 표현을 만나게 되므로 익혀두는 것이
> 중요합니다). 설명은 한국어입니다. 순서는 영어 알파벳순.

---

## 7.1 용어집

**Activation (활성값)** — 연산 사이를 흐르는 중간 값의 텐서(*가중치* 와 대비). VolvoxAI는 fp32로
유지합니다(네이티브 양자화 경로에서는 int8).

**Anchor (앵커)** — 탐지기가 박스를 처음부터 예측하는 대신 조정하는, 고정된 기준 박스(사전값).
EfficientDet-Lite0는 격자 셀당 9개 → 총 19,206개를 씁니다.

**Attention (어텐션, SDPA)** — 각 토큰이 자신의 **Query** 를 모든 토큰의 **Key** 와 비교하고 유사도로
그들의 **Value** 를 혼합하는 트랜스포머 메커니즘. "앞의 어떤 단어가 나에게 중요한가?"

**Autoregressive (자기회귀)** — 시퀀스를 한 번에 한 토큰씩 생성하며, 각 출력을 다시 입력으로 넣는 것.

**Backbone (백본)** — 비전 모델의 피처 추출 단계(여기서는 EfficientNet-Lite0).

**Backend (백엔드)** — 그래프 연산의 구체적 실행기. 브라우저 백엔드는 네 개의 *계층(tier)* 이고,
네이티브 백엔드는 CPU, Vulkan, OpenGL/GLES, Metal, NNAPI입니다. VolvoxAI는 노드마다 하나를 고릅니다.

**BiFPN** — Bi-directional Feature Pyramid Network. resize/pool/add로 해상도 간 피처를 융합해 모든
스케일이 디테일과 의미를 모두 갖게 합니다.

**BPE (바이트 페어 인코딩)** — 토크나이저 알고리즘: 바이트에서 시작해, 학습된 병합 목록에 따라 가장
빈번한 인접 쌍을 반복적으로 병합합니다.

**Broadcast (브로드캐스트)** — 원소별 연산에서 작은 텐서를 큰 텐서에 맞게 늘리는 것(예: 채널별 편향
`[C]` 를 `[N,H,W,C]` 에 더하기).

**Causal mask (인과적 마스크)** — 위치 *q* 가 `≤ q` 인 위치만 보도록 어텐션을 제한하는 것. 왼쪽→오른쪽
생성기를 만듭니다.

**Channel (채널)** — 텐서의 한 "피처 평면"(NHWC의 `C`). 입력 이미지는 3개(RGB), 은닉층은 여러 개를
가집니다.

**Convolution (합성곱, Conv2D)** — 작은 학습 필터를 이미지 위로 슬라이드하며 각 지점에서 내적해 패턴을
어디서나 감지. **뎁스와이즈** = 채널별 공간 필터, **포인트와이즈** = 1×1 채널 혼합기. 둘을 합친 것
(**뎁스와이즈 분리형**)이 저렴하며 백본을 구동합니다.

**Dequantize (역양자화)** — int8을 float로 되돌리기: `r = (q − zero_point) × scale`.

**dlopen / dlsym** — 공유 라이브러리를 로드하고 그 함수를 *실행 시점에* (링크 시점이 아니라) 찾는 것.
VolvoxAI의 네이티브 바이너리가 GPU SDK를 링크하지 않고 GPU 드라이버(`libvulkan`, `libGL`)를 쓰는
방식 — "정적 GPU 의존성 없음" 설계입니다.

**Embedding (임베딩)** — 이산 토큰을 나타내는 학습된 벡터. `Embedding` 연산은 표 조회입니다.

**Forward pass / Inference (순전파 / 추론)** — 그래프를 한 번 실행, 입력 → 출력. VolvoxAI는 이것만
합니다(학습 없음).

**Fusion (융합)** — 인접 연산을 병합해(예: Conv+ReLU) 중간 데이터를 한 번만 쓰는 것.

**GELU / ReLU / ReLU6 / Sigmoid** — 비선형성. 선형 층들 사이의 "결정 곡선". 이것들이 없으면 쌓인
MatMul이 하나로 붕괴합니다.

**GEMM** — GEneral Matrix Multiply(일반 행렬 곱). 대부분의 빠른 conv/matmul 경로가 귀결되는, 고도로
최적화된 루틴.

**Graph (그래프)** — 모델의 연산 목록: 이름 붙은 텐서로 연결된 노드(연산)들. `config.json` 으로 저장.

**Head (헤드)** — 최종 작업별 층: LM 헤드(→ 어휘 로짓) 또는 탐지기의 클래스/박스 헤드.

**im2col** — "image to columns". conv 입력 패치를 행렬로 펼쳐 conv를 GEMM으로 만드는 것.

**int8 / fp16 / fp32** — 8비트 정수 / 16비트 float / 32비트 float 숫자 형식(1 / 2 / 4 바이트).
4장 참고.

**KV-cache (KV-캐시)** — 지난 토큰들의 Key와 Value를 캐시해 각 생성 단계가 새 토큰의 어텐션만
계산하게 하는 것. 이 저장소에서는 `engine_prefill` + `engine_decode`.

**LayerNorm / RMSNorm** — 벡터를 정규화(평균 0, 분산 1, 그다음 학습된 스케일/이동)해 깊은 신경망의
숫자를 안정적으로 유지.

**Logits (로짓)** — 원시의, 정규화되지 않은 점수(소프트맥스/시그모이드 이전). LM 헤드와 클래스 헤드가
내놓습니다.

**MatMul (행렬 곱)** — 트랜스포머의 핵심 특징 혼합 연산.

**MBConv** — Mobile inverted BOTTLENECK conv 블록: 확장 → 뎁스와이즈 → 투영, 잔차 포함. 백본의 반복
단위입니다.

**NHWC / NCHW** — 텐서 차원 순서(배치, 높이, 너비, 채널) vs (배치, 채널, 높이, 너비). VolvoxAI 비전
모델은 NHWC를 씁니다.

**naga** — VolvoxAI가 하나의 WGSL 셰이더를 SPIR-V(Vulkan), GLSL(OpenGL), GLSL ES, MSL(Metal)로 교차
컴파일할 때 쓰는 Rust 도구. 생성된 셰이더 출력은 런타임 지원과 같지 않습니다. 해당 op를 그 백엔드에서
실행하려면 네이티브 디스패처가 백엔드 래퍼를 연결해야 합니다.

**NMS (비최대 억제, Non-Max Suppression)** — 겹치는 중복 탐지를 제거하고 객체당 점수가 가장 높은
박스를 남기는 후처리.

**NNAPI** — 안드로이드의 Neural Networks API. VolvoxAI의 네이티브 엔진이 안드로이드에서 이것으로
디스패치할 수 있습니다(`native/nnapi_engine.c`, `make build_android`).

**Node (노드)** — 그래프의 한 항목: 연산 + 그 입력/출력 텐서 이름과 파라미터.

**Op / Operation / Kernel (연산 / 커널)** — 하나의 수학 루틴(Add, Conv2D, SDPA…). "Op"은 그래프 수준의
이름, "kernel"은 그것의 구체적 구현입니다.

**Quantization (양자화)** — `scale` + `zero_point` 로 가중치/활성값을 더 적은 비트로 표현하는 것.

**Residual (잔차, skip connection)** — 블록의 입력을 출력에 더하는 것(`out = x + f(x)`). 정보와 기울기가
깊은 층을 지나 살아남게 합니다. 두 모델 모두에 있습니다.

**Safetensors** — 가중치의 표준 바이너리 파일 형식.

**Scale / Zero-point (스케일 / 제로포인트)** — 양자화 레시피의 두 숫자: 눈금 크기, 그리고 어떤 정수가
실제 0을 의미하는지.

**Softmax (소프트맥스)** — 점수 벡터를 확률 분포(양수, 합이 1)로 바꾸는 것.

**SPIR-V** — Vulkan이 소비하는 바이너리 셰이더 형식. `naga` 가 VolvoxAI의 WGSL을 여기로 컴파일합니다.

**Prefill / Decode (프리필 / 디코드)** — 네이티브 텍스트 생성의 두 단계: *prefill* 은 프롬프트를 한 번
실행해 KV-캐시를 채우고, *decode* 는 캐시를 써서 한 번에 새 토큰 하나를 실행합니다. `engine_prefill` /
`engine_decode` 참고.

**Tensor (텐서)** — 형태를 가진 다차원 숫자 배열. 엔진의 유일한 자료형입니다.

**Tier (계층)** — VolvoxAI의 네 브라우저 백엔드(WebNN / WebGPU / WASM / 순수 JS) 중 하나. 성능에 따라
선택됩니다.

**Token (토큰)** — 정수 id로 매핑된 텍스트 조각(단어/하위 단어/바이트).

**Weight (가중치)** — 학습 중에 얻어진 숫자. 추론 시 읽기 전용.

---

## 7.2 이 저장소를 지나는 경로

"개념은 이해했다"에서 "엔진을 수정할 수 있다"로 가려면 이 순서로 읽으세요:

1. **자료 모델** — `js/Tensor.js`(25줄), `js/Graph.js`(48줄). 작으니 전부 읽으세요.
2. **실행기** — `js/CPUEngine.js`. `for (node of graph.nodes)` 루프와 `switch` 디스패치를 보세요.
   이것이 런타임 전부입니다.
3. **네 개의 소박한 커널** — `js/ops/add.js`, `embedding.js`, `layerNorm.js`, `matMul.js`. 각각 몇십
   줄로 읽기 쉽습니다.
4. **두 모델의 설계도** — `models/tinystories_1m/config.json` 와
   `models/efficientdet_lite0_fp32/config.json` 을 훑고, 노드를 2–3장과 맞춰 보세요.
5. **어텐션 + conv 커널** — `js/ops/sDPA.js`, `js/ops/conv2D.js`. 두 개의 "심장".
6. **양자화** — `js/ops/dequantizeLinear.js`, 그다음 실제 int8 conv는 `native/quant_cpu_opt.c`.
7. **최적화** — `js/ops/conv2D.js` 를 `native/conv_f32_opt.c` 와 비교하며
   `docs/microkernel_optimization_guide.md` 와 `docs/xnnpack_optimization_guide.md` 를 읽으세요.
8. **GPU 계층** — `shaders/*.wgsl`(예: `matmul`)과 `js/GraphExecutor.js`.
9. **네이티브 엔진**(6장) — `native/engine.h` + `native/engine.c`(수명 주기),
   `native/engine_runtime.c`(`run_node` 백엔드 선택), 그다음 `native/vulkan_engine.c` 같은 기기 백엔드
   (맨 위의 `dlopen` 을 보세요). `native/main.c` 에 태스크 CLI가 있습니다.

`docs/operation_list.md` 는 연산×백엔드 지원 행렬입니다 — 당신의 참조 지도지요.

---

## 7.3 모델을 직접 실행해 보기

```bash
# 언어 모델 — 텍스트 생성(탐욕적). 먼저 네이티브 바이너리 빌드: `make build_native`.
./native/volvoxai generate models/tinystories_1m \
  --prompt "Once upon a time, Lily" --max-new 50 [--debug]

# 원시 그래프 러너 — 고정 토큰 집합에 대한 로짓 텐서 덤프.
./native/volvoxai run models/tinystories_1m \
  --input tokens=models/tinystories_1m/tokens.i32 \
  --input positions=models/tinystories_1m/positions.i32 \
  --output logits=out.f32 --last-token 4

# 객체 탐지기 — 이미지를 순위 박스로 디코드.
./native/volvoxai detect models/efficientdet_lite0_int8 \
  --image input0=photo.png --image-normalize raw-255 \
  --boxes boxes --scores scores --max-det 20

# Node에서(WASM / 순수 JS 계층), 아무 설계도나 스모크 테스트:
node bin/volvox.js run --model models/tinystories_1m/model.safetensors --backend wasm
```

`generate` 에 `--debug` 를 붙이면 노드별 시간과 초당 토큰(tokens/sec)을 볼 수 있습니다 — 시간이
어디로 가는지 *체감* 하기(그리고 5장의 최적화가 효과를 내는 것을 지켜보기)에 좋은 방법입니다.

---

## 7.4 실습 (읽기 → 실력)

1. **손으로 따라가기.** 시퀀스 `[5, 5]`(동일한 토큰 두 개)와 가상의 2차원 임베딩을 가정하세요.
   `Embedding → Add(위치) → LayerNorm` 을 종이와 펜으로 따라가고, 형태가 `config.json` 과 맞는지
   확인하세요.
2. **인과성 깨기.** `js/ops/sDPA.js` 에서 `k <= q` 를 `k < seq_len` 으로 바꾸세요. 생성 텍스트에 무슨
   일이 왜 일어날지 예측하세요. (그다음 되돌리세요.)
3. **가중치 양자화하기.** `scale = 0.02`, `zero_point = -5` 를 고르세요. `r = 0.31` 을 양자화한 뒤
   다시 역양자화하세요. 왕복 오차를 보고하세요. 이제 `scale = 0.002` 를 시도하세요. 정밀도가 범위에서
   무엇을 대가로 치렀나요?
4. **FLOP 세기.** 첫 `Conv2D`(stem: 320×320×3 → 160×160×32, 3×3 필터)의 곱-덧셈 수를 추정하세요. 같은
   출력 크기의 1×1 포인트와이즈 conv와 비교하세요. 왜 뎁스와이즈 분리형이 더 저렴한가요?
5. **연산 추가하기.** `js/ops/` 에 원소별 `Abs` 커널을 구현하고, `CPUEngine.js` 의 `switch` 에 연결한
   뒤, 디스패치되는지 확인하세요. (`js/ops/reLU.js` 를 템플릿으로 따라 하세요.)
6. **융합 찾기.** `models/efficientdet_lite0_fp32/config.json` 에서 `relu` 파라미터가 설정된 `Conv2D`
   를 찾으세요 — 그것이 이미 구워진 Conv+ReLU 융합입니다. 그것이 어떤 두 연산을 나타내는지 설명하세요.

---

## 7.5 이 코드베이스의 현재 빈틈

VolvoxAI는 추론 엔진이고, 이 교과서도 그 경계를 따릅니다. 이 코드베이스는 학습된 가중치를 로드하고
순전파를 실행할 수 있지만, 모델을 만들고 튜닝하고 과학적으로 검증하는 데 필요한 시스템은 아직 없습니다.

| 빠진 영역 | 추가되어야 할 것 | 왜 중요한가 |
|---|---|---|
| **학습** | reverse-mode autodiff, backward 커널, 손실 함수, Adam/SGD 같은 최적화기, 학습률 스케줄, 초기화, 체크포인팅, 정규화. | 가중치를 소비하는 대신 발견하는 방법입니다. |
| **수학 기초** | 선형대수 유도, 연쇄 법칙과 기울기를 위한 미적분, 확률, entropy/cross-entropy, KL divergence, likelihood. | 학습과 평가가 왜 그렇게 움직이는지 설명하는 도구입니다. |
| **데이터** | 데이터셋 manifest, 스트리밍/입력 파이프라인, 증강, 정제, 토크나이저 학습, train/validation/test 분할, 누수 점검. | 모델 품질은 보통 데이터 품질과 실험 위생에 의해 제한됩니다. |
| **평가와 실험** | 작업별 지표, 검증 루프, baseline, ablation, 하이퍼파라미터 sweep, 과적합 점검, bias-variance 분석. | 모델이 실제로 더 좋아졌는지, 단지 달라졌는지 구분하는 방법입니다. |
| **아키텍처 폭** | diffusion, graph neural network, RNN/LSTM, 강화학습, VAE/GAN, retrieval과 embedding 시스템, multimodal 모델, mixture-of-experts, state-space model. | 현재 설명은 트랜스포머 LM 하나와 CNN 탐지기 하나입니다. 많은 도메인은 다른 inductive bias를 씁니다. |
| **최신 LLM 학습 스택** | 사전학습 루프, 지도 미세조정, LoRA/adapter, RLHF/DPO 선호 학습, 분산 데이터/모델 병렬화, FlashAttention 내부. | 대규모 LLM 작업의 대부분은 학습 레시피, 메모리 효율적 어텐션, 대규모 시스템 주변에서 일어납니다. |
| **연구 실무** | 논문 재현, 결과 유도, 통제 실험, scaling-law 분석, 오류 분석, 가정 문서화. | 모델을 실행하는 것과 신뢰할 수 있는 새 지식을 만드는 것의 차이입니다. |

이것들은 추론 장들의 선행 조건이 아니라, 앞으로 추가될 수 있는 교과서/코드 모듈입니다. 범위는 명확합니다:
이 저장소는 강한 추론/런타임 기초이지만, 완전한 학습·연구 커리큘럼은 아닙니다.

---

## 7.6 여기서 어디로 갈까

자연스러운 다음 단계는 두 갈래로 나뉩니다:

- **이 저장소의 추론 경로를 더 깊게 파기.** `native/conv_f32_opt.c` 를 Google의 **XNNPACK** 과
  비교하세요. 이 저장소의 `docs/xnnpack_optimization_guide.md` 가 안내된 투어입니다. 그다음
  `shaders/*.wgsl` 과 네이티브 GPU 백엔드를 살펴보세요.
- **트랜스포머 키우기.** GPT-2/3, LLaMA, Mistral, Qwen은 2장의 그래프를 더 넓고 깊게 한 것에 약간의
  변형을 더한 것입니다: LayerNorm 대신 **RMSNorm**, 학습된 `wpe` 대신 **RoPE** 회전 위치,
  **그룹 쿼리 어텐션(grouped-query attention)**, **SwiGLU** MLP. 각각은 당신이 아는 연산의 작은
  변주입니다.
- **아키텍처 넓히기.** 분류, 분할, 자세, diffusion, retrieval, multimodal, MoE, SSM 시스템은 모두
  텐서/그래프 멘탈 모델을 재사용하지만, 서로 다른 블록과 학습 목표를 더합니다.
- **학습 빈틈을 의도적으로 메우기.** 작은 autograd 엔진, cross-entropy 손실, Adam, 작은 데이터셋 로더,
  검증 루프가 첫 번째 구체적 추가가 될 수 있습니다.
- **원 논문 읽기**, 메커니즘이 구체화된 뒤: *Attention Is All You Need*(트랜스포머),
  *EfficientDet*(이 탐지기), *EfficientNet*(백본), 그리고 양자화 입문(예: "gemmlowp"/TFLite 정수
  양자화 글).
- **이웃 런타임 탐색하기**: **wonnx**(WebGPU/ONNX), **ncnn**(무의존성 네이티브),
  **ggml/llama.cpp**(이식성 C LLM 추론).

여기서 세운 멘탈 모델 — *모델은 학습된 가중치를 가진 작은 텐서 연산의 그래프다. 추론은 그래프를
훑는다. 성능은 메모리 배치다. 정밀도는 크기/정확도 다이얼이다* — 은 앞으로의 모듈들에도 전이되지만,
그것은 전체 스택의 한 부분입니다.

---

*VolvoxAI 교과서의 끝. [목차](README.md)로 돌아가기.*
