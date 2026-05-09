# Requirements Document

## Introduction

본 문서는 Qiu et al. 2025 ("Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models") 논문의 Edge LLM 복제·확장 연구에서 발견된 문제를 수정하고, "태스크 복잡도에 따른 Bayesian/Oracle teaching 효과" 서사로 연구를 피벗하기 위한 요구사항을 정의한다.

현재 상태:
- Gemma 4 E2B (2.3B params) + QLoRA로 2-feature 항공편 추천 태스크를 학습
- 2-feature 평가 결과: Base(Δ=-0.010), Bayesian(Δ=-0.014), Oracle(Δ=-0.032) — 모든 조건에서 학습 효과 없음
- 근본 원인 2가지 진단 완료: (1) 평가 코드의 비대화형 피드백 문제, (2) E2B LoRA alpha 보수적 설정

연구 피벗 방향:
- 2-feature(간단) → 3-feature(중간) → 4-feature(스트레스 테스트) 복잡도별 비교
- Bayesian vs Oracle teaching 효과 비교
- 2/3-feature 긍정 결과를 논문 핵심으로, 4-feature를 한계 분석으로 구성

## Glossary

- **Evaluator**: evaluation_paper_style.py 모듈. Interaction_Data로 학습 컨텍스트를 구성하고 Heldout_Data로 정확도를 측정한다
- **Trainer**: train.py 모듈. QLoRA 기반으로 Gemma 4 모델을 파인튜닝한다
- **Data_Generator**: data_generation_nfeat.py 모듈. N-feature 도메인의 Bayesian/Oracle 학습 데이터를 생성한다
- **Interactive_Evaluation**: 모델이 각 라운드에서 실제로 예측하고, 예측 결과에 기반한 correct/incorrect 피드백을 받는 평가 방식. 원 논문의 평가 방식과 동일하다
- **Non_Interactive_Evaluation**: 학습 컨텍스트에 gold label을 미리 채워넣고 항상 "correct" 피드백을 사용하는 현재의 평가 방식. 모델이 실제로 추론하지 않는다
- **Interaction_Data**: 평가 시 학습 컨텍스트로 사용되는 사용자별 5라운드 대화 데이터 (flight_Nfeatures.jsonl)
- **Heldout_Data**: 평가용 100개 항공편 세트와 사용자별 정답 인덱스 (flight_Nfeatures.json)
- **Train_Data**: 모델 파인튜닝에 사용되는 2000명 × 5라운드 대화 데이터 (train_Nfeat_teaching.jsonl)
- **Round_Accuracy**: 특정 라운드에서 Heldout_Data 100 sets에 대한 정답률
- **Learning_Effect**: R5(Round 5 정확도) - R1(Round 1 정확도)로 측정되는 학습 개선폭
- **Task_Complexity**: 항공편 선택에 사용되는 feature 수. 2feat(duration, price), 3feat(price, departure_time, duration), 4feat(departure_time, duration, stops, price)
- **LoRA_Alpha**: Low-Rank Adaptation의 스케일링 계수. alpha/rank 비율이 어댑터 업데이트 강도를 결정한다
- **Figure_Generator**: 논문용 그래프 및 LaTeX 표를 생성하는 시각화 모듈
- **Pipeline**: 데이터 생성 → 학습 → 평가 → 결과 분석의 전체 실험 워크플로우
- **SLURM_Orchestrator**: SLURM 클러스터에서 학습·평가 작업을 자동 제출하고 의존성을 관리하는 스크립트

## Requirements

### Requirement 1: Interactive Evaluation 구현

**User Story:** 연구자로서, 원 논문과 동일한 대화형 평가 방식을 구현하여 모델의 실제 학습 효과를 정확히 측정하고 싶다.

#### Acceptance Criteria

1. WHEN 평가가 실행되면, THE Evaluator SHALL 각 학습 라운드에서 모델이 실제로 항공편을 예측하도록 추론을 수행한다
2. WHEN 모델이 라운드 R에서 Flight X를 예측하고 사용자의 실제 선택이 Flight Y이면, THE Evaluator SHALL 라운드 R+1의 피드백으로 "Your option Flight X is incorrect. I prefer Flight Y."를 사용한다
3. WHEN 모델이 라운드 R에서 사용자의 실제 선택과 동일한 Flight X를 예측하면, THE Evaluator SHALL 라운드 R+1의 피드백으로 "Your option Flight X is correct."를 사용한다
4. THE Evaluator SHALL 학습 컨텍스트의 assistant 응답으로 gold label 대신 모델의 실제 예측 결과를 사용한다
5. THE Evaluator SHALL Non_Interactive_Evaluation 모드를 기존 코드와의 호환성을 위해 CLI 플래그(--interactive)로 선택 가능하게 유지한다
6. WHEN Interactive_Evaluation에서 모델 응답 파싱이 실패하면, THE Evaluator SHALL 해당 라운드를 랜덤 선택(uniform)으로 대체하고 파싱 실패 횟수를 별도로 기록한다

### Requirement 2: E2B LoRA Alpha 수정 및 재학습

**User Story:** 연구자로서, E2B 모델의 LoRA alpha를 표준 권장값으로 수정하여 어댑터 업데이트 강도를 적절히 설정하고 싶다.

#### Acceptance Criteria

1. THE Trainer SHALL E2B 모델의 LoRA alpha를 16에서 32로 변경한다 (lora_r=16, lora_alpha=32, 비율 2.0)
2. WHEN E2B 2-feature 모델 재학습이 완료되면, THE Trainer SHALL Bayesian 및 Oracle 두 조건 모두에 대해 재학습을 수행한다
3. THE Trainer SHALL 재학습된 모델의 train_info.json에 변경된 lora_alpha 값을 기록한다
4. THE Trainer SHALL 기존 모델(alpha=16)을 덮어쓰지 않고 별도 디렉토리에 저장하여 비교를 가능하게 한다

### Requirement 3: 3-Feature 학습 데이터 생성

**User Story:** 연구자로서, 3-feature 도메인의 학습 데이터를 생성하여 중간 복잡도 실험을 수행하고 싶다.

#### Acceptance Criteria

1. WHEN 3-feature 데이터 생성이 요청되면, THE Data_Generator SHALL 2000명 사용자 × 5라운드의 Bayesian 및 Oracle Train_Data를 생성한다
2. THE Data_Generator SHALL 3-feature Train_Data의 항공편 텍스트 형식(feature 순서: price, departure_time, duration)을 flight_3features.jsonl의 Interaction_Data 형식과 동일하게 생성한다
3. THE Data_Generator SHALL 3-feature Train_Data에서 27개 사용자 타입(3^3 weight 조합) 전체를 균등하게 샘플링한다
4. THE Data_Generator SHALL 생성된 3-feature Train_Data의 Bayesian_Assistant 예측 정확도(Oracle 대비)를 라운드별로 출력한다

### Requirement 4: 복잡도별 체계적 실험 수행

**User Story:** 연구자로서, 2/3/4-feature 복잡도에 따른 Bayesian 및 Oracle teaching 효과를 체계적으로 비교하고 싶다.

#### Acceptance Criteria

1. THE Pipeline SHALL 2-feature, 3-feature, 4-feature 각각에 대해 Base, Bayesian, Oracle 조건의 Interactive_Evaluation을 수행한다 (총 9개 실험 조건)
2. THE Evaluator SHALL 각 실험 조건에 대해 R1~R5 Round_Accuracy와 Learning_Effect를 계산한다
3. THE Evaluator SHALL 각 실험 조건에 대해 사용자별 정확도 분포(평균, 표준편차)를 보고한다
4. THE Pipeline SHALL 동일한 하이퍼파라미터 설정(lora_r=16, lora_alpha=32, lr=1e-4)으로 모든 E2B 복잡도 실험을 수행한다
5. WHEN 평가 결과의 파싱 실패율이 5%를 초과하면, THE Evaluator SHALL 경고 메시지를 출력한다
6. THE Pipeline SHALL 모든 실험 결과를 results/paper_style/ 디렉토리에 {model}\_{teaching}\_{quant}\_{domain}\_n{users}\_s{sets}.json 형식으로 저장한다

### Requirement 5: SLURM 작업 자동화

**User Story:** 연구자로서, 9개 실험 조건의 학습·평가 작업을 SLURM 클러스터에서 효율적으로 관리하고 싶다.

#### Acceptance Criteria

1. THE SLURM_Orchestrator SHALL 각 실험 조건(복잡도 × 교육방식)에 대한 학습 및 평가 SLURM 작업 스크립트를 생성한다
2. THE SLURM_Orchestrator SHALL 학습 완료 후 평가가 실행되도록 SLURM job dependency(--dependency=afterok)를 설정한다
3. THE SLURM_Orchestrator SHALL 모든 작업의 stdout/stderr 로그를 logs/ 디렉토리에 {task}\_{domain}\_{teaching}\_{jobid}.out 형식으로 저장한다
4. THE SLURM_Orchestrator SHALL RTX A6000 GPU 1개, 메모리 96GB(학습) 또는 64GB(평가)를 요청한다

### Requirement 6: 논문용 시각화 생성

**User Story:** 연구자로서, 실험 결과를 논문에 포함할 출판 품질의 그래프와 표로 자동 생성하고 싶다.

#### Acceptance Criteria

1. WHEN 시각화가 요청되면, THE Figure_Generator SHALL 복잡도별(2/3/4-feature) × 교육방식별(Base/Bayesian/Oracle) 라운드 진행에 따른 정확도 변화 그래프를 생성한다
2. THE Figure_Generator SHALL 모든 그래프를 300 DPI 이상의 해상도로 PDF 및 PNG 형식으로 저장한다
3. THE Figure_Generator SHALL 복잡도 × 교육방식의 주요 지표(R1, R5, Learning_Effect)를 요약하는 LaTeX 표를 생성한다
4. THE Figure_Generator SHALL 그래프 내부 레이블을 영문으로, 캡션 텍스트를 한글로 생성한다

### Requirement 7: 결과 종합 분석

**User Story:** 연구자로서, 모든 실험 결과를 종합 분석하여 "태스크 복잡도에 따른 교육 효과" 서사를 뒷받침하는 통계적 근거를 확보하고 싶다.

#### Acceptance Criteria

1. WHEN 종합 분석이 요청되면, THE Pipeline SHALL 복잡도별 Learning_Effect의 통계적 유의성을 paired t-test 또는 Wilcoxon signed-rank test로 검정한다
2. WHEN 종합 분석이 요청되면, THE Pipeline SHALL 복잡도 증가에 따른 Learning_Effect 감소 추세를 정량적으로 보고한다
3. WHEN 종합 분석이 요청되면, THE Pipeline SHALL Bayesian vs Oracle teaching의 복잡도별 성능 차이를 보고한다
4. IF 2-feature에서도 유의미한 Learning_Effect가 관찰되지 않으면, THEN THE Pipeline SHALL 추가 진단 항목(학습 데이터 품질, 모델 응답 형식 변화, 추가 하이퍼파라미터 탐색 필요성)을 보고한다
