# Requirements Document

## Introduction

본 문서는 Qiu et al. 2025 ("Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models")의 main finding을 Gemma 2 9B 기반으로 재현 완료한 결과를 출발점으로, **Bayesian teaching 기법의 두 가지 응용 방향**을 검증하기 위한 요구사항을 정의한다.

### 연구 배경 (이미 완료된 부분)

원 논문 핵심 수식:
- **Eq. 1 (User selection):** o*(O,θ) = argmax_o θᵀφ(o)
- **Eq. 2 (Posterior update):** q^(i+1)(θ) = p(o*|θ,O) · q^(i)(θ) / p(o*|O)
- **Eq. 3 (Likelihood):** p(o*|θ,O) = 𝟙[argmax_o θᵀφ(o) = o*]
- **Bayesian assistant prediction:** θ̂ = E_q[θ], ô = argmax_o θ̂ᵀφ(o)

Gemma 2 9B 재현 결과:
- Bayesian(bf16): R1=38.5% → R5=70.8%, Learning_Effect=+0.324
- Oracle(bf16):   R1=37.5% → R5=57.4%, Learning_Effect=+0.199
- 결론: Bayesian > Oracle 재현됨 ✓

기 구축 인프라:
- 학습된 모델: `models/gemma2-9b-bayesian/`, `models/gemma2-9b-oracle/`
- Inference 모듈: `src/inference.py` (bf16/int8/int4 BitsAndBytesConfig 지원)
- 평가 모듈: `src/evaluation.py` (reward_fn 기반 ground-truth 직접 계산)
- 평가 스크립트: `scripts/evaluate.py` (`--quantization` 플래그 지원)
- 평가 데이터: `data/eval/interaction/flight.jsonl`, `data/eval/heldout/flight.json`, `data/eval/heldout/hotel.json`
- 기존 결과: `results/gemma2-9b-{bayesian,oracle}_{teaching}_bf16_flight.json` (per-round / per-user 데이터 포함)

### 연구 응용 방향

**Application C — Quantization Robustness:**
"Bayesian teaching으로 학습된 모델은 INT8/INT4 양자화에 더 강건한가?"
가설: Oracle은 학습 데이터 패턴을 단순 암기했지만 Bayesian은 사후분포 업데이트 추론 과정을 학습했으므로 양자화에 따른 성능 저하가 더 작을 것이다.

**Application D — Sample Efficiency:**
"Bayesian teaching이 Oracle보다 더 적은 라운드(샘플)로 사용자 선호를 추론하는가?"
가설: Bayesian은 명시적 사후분포 추적을 학습했으므로 라운드 진행에 따른 학습 곡선의 초기 기울기가 더 가파르며, 임계 정확도에 더 빠르게 도달한다.

### 산출물

4페이지 한국어 단편 논문 (구성: Introduction & Background → Method → Experiments → Results & Discussion).

## Glossary

- **Pipeline:** 양자화 평가 → 결과 분석 → 시각화 → 통계 검정의 전체 실험 워크플로우
- **Quant_Evaluator:** `scripts/evaluate.py` 기반의 양자화 평가 실행 모듈. `--quantization {bf16,int8,int4}` × `--teaching {bayesian,oracle}` × `--domain {flight,hotel}` 조합으로 호출된다
- **Inference_Engine:** `src/inference.py`의 `GemmaInference` 클래스. BitsAndBytesConfig로 정밀도를 결정하고 GPU 메모리/지연시간을 측정한다
- **Round_Accuracy:** 특정 라운드 r ∈ {1..5}에서 Heldout 사용자 집합 전체의 정답률
- **Learning_Effect (LE):** R5 - R1 (라운드 5 정확도 - 라운드 1 정확도)
- **Per_User_Round_Trace:** 사용자 u에 대해 각 라운드 r에서의 (정답 여부, 예측, ground truth)를 기록한 배열
- **Quantization_Condition:** {bf16, int8, int4} 중 하나의 정밀도 설정
- **Teaching_Condition:** {bayesian, oracle} 중 하나의 학습 방식
- **Experiment_Condition:** (Teaching_Condition × Quantization_Condition × Domain) 의 단일 조합
- **Robustness_Drop_R5 (Δ_R5):** R5_bf16 - R5_quant (양수면 양자화로 인해 정확도 하락). int8과 int4 각각에 대해 정의됨
- **Robustness_Drop_LE (Δ_LE):** LE_bf16 - LE_quant
- **Robustness_Gap:** Bayesian과 Oracle 사이의 Robustness_Drop 차이. (Δ_LE_oracle - Δ_LE_bayesian)이 양수이면 Bayesian이 더 강건함을 의미한다
- **Convergence_Round:** 사용자별로 R5 정확도 수준의 80%(상대 임계값) 또는 절대 임계값(예: 정답 누적 ≥ 임계)에 도달하는 최초 라운드
- **Learning_Curve:** 라운드(x축) 대 누적/순간 정확도(y축) 곡선
- **Memory_Footprint:** 모델 로드 후 측정된 GPU 메모리 사용량 (allocated / reserved MiB)
- **Inference_Latency:** 단일 사용자 5라운드 평가에 소요된 시간 (초). 또는 토큰당 평균 생성 시간
- **Sample_Efficiency_Score:** 라운드 r ≤ 3 구간에서의 학습 곡선 기울기 (LE_partial = R3 - R1) 또는 AUC(라운드별 정확도의 평균)
- **Result_File:** Quant_Evaluator의 출력 JSON. 파일명 형식 `{model_slug}_{teaching}_{quant}_{domain}.json`
- **Cross_Domain_Evaluation:** Flight 도메인으로 학습된 모델을 Hotel 도메인 Heldout 데이터에서 평가하는 OOD 평가
- **Figure_Generator:** matplotlib 기반의 그래프/표 생성 모듈
- **Master_Script:** 모든 실험 조건을 일관된 시드와 설정으로 순차/병렬 실행하는 단일 진입점 스크립트

## Requirements

### Requirement 1: 양자화 평가 매트릭스 실행

**User Story:** 연구자로서, Bayesian과 Oracle 두 모델을 bf16/int8/int4 세 가지 정밀도에서 일관된 방식으로 평가하여 양자화 robustness를 비교하고 싶다.

#### Acceptance Criteria

1. THE Pipeline SHALL `models/gemma2-9b-bayesian` 와 `models/gemma2-9b-oracle` 두 모델 각각에 대해 bf16, int8, int4 세 가지 Quantization_Condition으로 Flight 도메인 평가를 수행한다 (총 6개 Experiment_Condition).
2. WHEN 한 Experiment_Condition의 평가가 시작되면, THE Quant_Evaluator SHALL `scripts/evaluate.py`를 호출하고 동일한 `--num-rounds=5`, 동일한 사용자 집합, 동일한 random seed를 사용한다.
3. THE Quant_Evaluator SHALL 각 Experiment_Condition의 결과를 `results/quant_efficiency/{model_slug}_{teaching}_{quant}_flight.json` 경로에 저장한다.
4. WHERE 기존 `results/gemma2-9b-{bayesian,oracle}_{teaching}_bf16_flight.json` 파일이 존재하면, THE Pipeline SHALL 해당 파일을 그대로 재사용하고 bf16 평가를 다시 실행하지 않는다.
5. THE Result_File SHALL `summary.round_accuracies` (길이 5 배열), `summary.learning_effect`, `summary.parse_failure_rate`, `per_user` (사용자별 라운드 trace), `memory_mib`, `elapsed_seconds_total` 필드를 포함한다.
6. IF 한 Experiment_Condition의 `summary.parse_failure_rate` 가 0.05를 초과하면, THEN THE Pipeline SHALL 경고 메시지를 stdout에 출력하고 해당 조건을 결과 매니페스트에 `flagged=true` 로 표시한다.

### Requirement 2: 메모리 및 지연시간 측정

**User Story:** 연구자로서, 양자화에 따른 GPU 메모리 절감 및 지연시간 변화를 함께 측정하여 정확도-효율성 트레이드오프를 정량화하고 싶다.

#### Acceptance Criteria

1. WHEN 한 Experiment_Condition 평가가 완료되면, THE Inference_Engine SHALL 모델 로드 직후의 `torch.cuda.memory_allocated` 와 `torch.cuda.memory_reserved` 를 MiB 단위로 기록한다.
2. THE Quant_Evaluator SHALL 평가 전체 wall-clock 시간 (`elapsed_seconds_total`) 과 사용자당 평균 시간 (`elapsed_seconds_total / num_users`) 을 Result_File에 기록한다.
3. THE Quant_Evaluator SHALL 라운드당 평균 응답 생성 토큰 수 (또는 평균 생성 시간) 를 추정하기 위해 평가 시작·종료 wall-clock과 생성된 응답 길이를 수집하여 Result_File의 `latency` 서브 객체에 저장하며, 응답 길이가 0인 라운드의 경우에도 wall-clock 시간 기록은 누락 없이 포함한다.
4. THE Pipeline SHALL 6개 Experiment_Condition 전체에 대한 (Quantization_Condition × Memory_Footprint × Inference_Latency) 표를 `results/quant_efficiency/efficiency_table.csv` 로 출력한다.
5. WHEN GPU 메모리 측정이 실행되면, THE Inference_Engine SHALL 측정 직전 `torch.cuda.empty_cache()` 와 `torch.cuda.reset_peak_memory_stats()` 를 호출하여 측정 일관성을 보장한다.

### Requirement 3: Sample Efficiency 분석

**User Story:** 연구자로서, 기존 평가 결과 JSON으로부터 라운드별 학습 곡선을 추출하여 Bayesian의 라운드 효율성 우위를 정량적으로 보이고 싶다.

#### Acceptance Criteria

1. WHEN Sample Efficiency 분석이 요청되면, THE Pipeline SHALL `results/quant_efficiency/` 와 `results/` 의 모든 Result_File 을 로드하여 (Teaching_Condition × Quantization_Condition × Domain) 별로 집계한다.
2. THE Pipeline SHALL 각 Experiment_Condition에 대해 R1~R5 Round_Accuracy 시계열, Learning_Effect (R5-R1), Sample_Efficiency_Score (R3-R1), 그리고 라운드별 누적 정확도 AUC (사다리꼴 적분) 를 계산한다.
3. THE Pipeline SHALL 각 사용자 u에 대해 Convergence_Round 를 다음 두 정의로 계산한다: (a) 해당 Experiment_Condition의 R5 정확도 × 0.8 임계 도달 라운드, (b) 누적 정답 수가 3 이상이 되는 최초 라운드. 임계 미도달 사용자는 라운드 6 (right-censored) 로 표시한다.
4. THE Pipeline SHALL Per_User_Round_Trace 로부터 사용자별 학습 곡선의 분산을 계산하고 (Teaching_Condition × Quantization_Condition) 별로 mean ± std 를 보고한다.
5. THE Pipeline SHALL 분석 결과를 `results/quant_efficiency/sample_efficiency_summary.json` 으로 저장하며, 이 파일은 각 Experiment_Condition에 대해 `{R1, R2, R3, R4, R5, LE, sample_efficiency_score, AUC, convergence_round_mean, convergence_round_median, learning_curve_std}` 필드를 포함한다.

### Requirement 4: Robustness 지표 정의 및 산출

**User Story:** 연구자로서, 양자화에 따른 Bayesian과 Oracle의 성능 저하를 동일 지표로 비교하여 Robustness_Gap을 정량화하고 싶다.

#### Acceptance Criteria

1. WHEN Robustness 분석이 요청되면, THE Pipeline SHALL 각 Teaching_Condition (bayesian, oracle) 에 대해 Δ_R5(int8) = R5_bf16 - R5_int8, Δ_R5(int4) = R5_bf16 - R5_int4 를 계산한다.
2. THE Pipeline SHALL 각 Teaching_Condition에 대해 Δ_LE(int8) = LE_bf16 - LE_int8, Δ_LE(int4) = LE_bf16 - LE_int4 를 계산한다.
3. THE Pipeline SHALL Robustness_Gap 을 (Δ_R5_oracle - Δ_R5_bayesian) 와 (Δ_LE_oracle - Δ_LE_bayesian) 두 형태로 보고하며, int8 및 int4 각각에 대해 별도로 계산한다.
4. THE Pipeline SHALL Robustness 지표를 `results/quant_efficiency/robustness_summary.json` 으로 저장하고, 동시에 `results/quant_efficiency/robustness_table.csv` 로 (teaching, quant, R1, R5, LE, Δ_R5, Δ_LE) 형태의 표를 출력한다.
5. THE Pipeline SHALL 각 사용자 u에 대해 per-user Δ_LE 를 계산하여 분포를 보고한다 (mean, median, IQR).

### Requirement 5: 통계적 유의성 검정

**User Story:** 연구자로서, Bayesian과 Oracle 간 robustness 차이가 통계적으로 유의한지 검증하여 결론의 신뢰도를 확보하고 싶다.

#### Acceptance Criteria

1. WHEN 통계 검정이 요청되면, THE Pipeline SHALL 동일한 사용자 인덱스를 가진 Bayesian과 Oracle 간 per-user Δ_LE(int8) 와 Δ_LE(int4) 차이에 대해 paired Wilcoxon signed-rank test 를 수행한다.
2. THE Pipeline SHALL 각 검정에 대해 검정 통계량, p-value, 효과 크기 (rank-biserial correlation 또는 Cohen's d) 를 보고한다.
3. WHEN 정규성을 가정할 수 있는 경우, THE Pipeline SHALL paired t-test 결과를 보조 지표로 함께 보고한다.
4. THE Pipeline SHALL 모든 검정 결과를 `results/quant_efficiency/significance_tests.json` 으로 저장하며, 각 검정에 대해 `{test_name, statistic, p_value, n_pairs, effect_size, interpretation}` 필드를 포함한다.
5. WHEN p-value < 0.05 인 검정 결과가 발견되면, THE Pipeline SHALL stdout에 해당 검정명과 결론을 강조 출력한다.

### Requirement 6: 시각화 및 LaTeX 표 생성

**User Story:** 연구자로서, 4페이지 한국어 논문에 포함할 출판 품질의 그래프와 표를 자동 생성하고 싶다.

#### Acceptance Criteria

1. WHEN 시각화가 요청되면, THE Figure_Generator SHALL 라운드(x: 1..5)별 Round_Accuracy 곡선을 (Teaching_Condition × Quantization_Condition) 별로 한 패널에 6개 선으로 그린 그래프를 생성한다.
2. THE Figure_Generator SHALL Quantization_Condition을 x축으로, R5 와 Learning_Effect를 y축으로 하는 막대그래프를 Bayesian 과 Oracle을 색으로 구분하여 생성한다.
3. THE Figure_Generator SHALL per-user Sample_Efficiency_Score (Bayesian) 대 per-user Sample_Efficiency_Score (Oracle) 산점도를 사용자 인덱스를 매칭하여 생성한다.
4. THE Figure_Generator SHALL Acceptance Criteria 6.1, 6.2, 6.3 의 세 그래프가 모두 메모리에 생성 완료된 이후에만 디스크 저장 단계를 시작하며, PDF 와 PNG (300 DPI) 두 형식으로 `results/quant_efficiency/figures/` 에 저장한다.
5. THE Figure_Generator SHALL 그래프 내부 레이블 (축, 범례, 제목) 을 영문으로 작성하며, 별도의 캡션 텍스트는 한글로 `results/quant_efficiency/figures/captions.md` 에 저장한다.
6. THE Figure_Generator SHALL Robustness 요약표 (rows: Teaching × Quant, cols: R1, R5, LE, Δ_R5, Δ_LE) 와 Sample_Efficiency 요약표 (rows: Teaching × Quant, cols: R3-R1, AUC, Convergence_Round) 를 `results/quant_efficiency/tables/*.tex` 로 출력한다 (booktabs 포맷).

### Requirement 7: 재현성 및 마스터 스크립트

**User Story:** 연구자로서, 단일 명령으로 전체 파이프라인을 재현 가능하게 실행하고, 모든 산출물의 경로를 표준화하고 싶다.

#### Acceptance Criteria

1. THE Master_Script SHALL `scripts/run_quant_efficiency.py` 라는 단일 진입점으로 제공되며, `--stage {evaluate,analyze,visualize,all}` 플래그로 실행 단계를 선택할 수 있다.
2. WHEN `--stage evaluate` 가 지정되면, THE Master_Script SHALL Requirement 1 의 6개 Experiment_Condition 평가를 순차 실행한다 (GPU 메모리 충돌 방지).
3. WHEN `--stage analyze` 가 지정되면, THE Master_Script SHALL Requirement 3, 4, 5 의 분석 단계를 실행한다.
4. WHEN `--stage visualize` 가 지정되면, THE Master_Script SHALL Requirement 6 의 시각화 단계를 실행한다.
5. THE Master_Script SHALL 모든 평가 호출에 대해 동일한 random seed (기본 42) 를 환경변수 `PYTHONHASHSEED`, `transformers.set_seed`, `torch.manual_seed` 에 설정한다.
6. THE Master_Script SHALL 실행 시작 시점에 `results/quant_efficiency/run_manifest.json` 을 생성하며, 이 파일은 `{timestamp, git_commit_sha, python_version, torch_version, transformers_version, bnb_version, gpu_name, seed, conditions: [...]}` 를 포함한다.
7. WHEN 한 Experiment_Condition 의 결과 파일이 이미 존재하면, THE Master_Script SHALL 해당 조건을 어떠한 처리·로깅·매니페스트 갱신 없이 완전히 건너뛰며, `--force` 플래그가 지정된 경우에만 재실행한다.

### Requirement 8: Cross-Domain Robustness (선택적)

**User Story:** 연구자로서, Flight 도메인으로 학습된 Bayesian/Oracle 모델이 Hotel 도메인에서도 robustness 우위를 유지하는지 (선택적으로) 검증하고 싶다.

#### Acceptance Criteria

1. WHERE `--include-cross-domain` 플래그가 지정되면, THE Pipeline SHALL Bayesian 과 Oracle 모델 각각에 대해 bf16/int8/int4 세 정밀도로 Hotel 도메인 평가를 수행한다 (추가 6개 Experiment_Condition).
2. WHERE Cross_Domain_Evaluation 이 수행되면, THE Pipeline SHALL Flight 와 Hotel 두 도메인의 Robustness 지표 (Δ_R5, Δ_LE) 를 비교 표로 출력한다.
3. WHERE Cross_Domain_Evaluation 결과가 존재하면, THE Figure_Generator SHALL Flight 대 Hotel 도메인의 Robustness 비교 그래프를 추가 생성한다.
4. WHERE Cross_Domain_Evaluation 이 실제 수행될 때에 한해, THE Pipeline SHALL Cross_Domain_Evaluation 결과 파일을 `results/quant_efficiency/{model_slug}_{teaching}_{quant}_hotel.json` 경로에 저장하며, 평가가 수행되지 않은 경우 어떤 placeholder 파일도 생성하지 않는다.
5. WHERE `--include-cross-domain` 이 지정되지 않으면, THE Pipeline SHALL Hotel 도메인 평가 단계를 건너뛰고 Flight 도메인 결과만 산출한다.
