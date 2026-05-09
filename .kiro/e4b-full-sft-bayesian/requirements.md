# Requirements Document

## Introduction

본 문서는 Qiu et al. 2025 ("Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models", Nature Communications) 논문을 Gemma 4 E4B-it (8.08B params, 4.5B effective) 모델에서 Full SFT로 복제·확장하기 위한 요구사항을 정의한다.

이전 접근(E2B + QLoRA + Unsloth)이 실패한 근본 원인:
1. QLoRA는 소형 모델의 reasoning task에 불충분 (파라미터 업데이트 비율 < 1%)
2. Unsloth 프레임워크의 모델 로딩 호환성 문제
3. 학습 데이터 분포 불일치 (2/3-feature 실험)

새로운 접근:
- **모델**: Gemma 4 E4B-it (8.08B total, 4.5B effective) — Google의 최신 edge 모델 (2026.04)
- **학습**: Full SFT (전체 파라미터) — alignment-handbook + Accelerate + DeepSpeed ZeRO-3
- **하드웨어**: 6× NVIDIA RTX A6000 (48GB × 6 = 288GB) on SLURM (partition p02)
- **데이터**: 원 논문 학습 데이터 (bayesian.jsonl + oracle.jsonl, 31,200 records, 4-feature flight domain)
- **평가**: Interactive evaluation, 2/3/4/5-feature 도메인 일반화 테스트
- **후처리**: 학습된 모델의 양자화(4-bit, 8-bit) 후 성능 유지율 측정

연구 기여:
1. Gemma 4 (2026.04 출시)에 대한 최초 Bayesian teaching 실험
2. Edge 지향 모델(E4B, 8B params)에서의 Full SFT
3. 학습 후 양자화 영향 분석 (bf16 → int8 → int4)
4. 태스크 복잡도별 일반화 (2~5 features)
5. Bayesian vs Oracle teaching 비교 (최신 아키텍처)

## Glossary

- **Full_SFT_Trainer**: alignment-handbook의 run_sft.py 기반 학습 모듈. DeepSpeed ZeRO-3로 모델 전체 파라미터를 파인튜닝한다
- **DeepSpeed_ZeRO3**: 모델 파라미터, 그래디언트, 옵티마이저 상태를 GPU 간 분산하여 대형 모델의 Full SFT를 가능하게 하는 분산 학습 전략
- **Accelerate**: Hugging Face의 분산 학습 런처. DeepSpeed, FSDP 등 다양한 백엔드를 통합 관리한다
- **Alignment_Handbook**: Hugging Face의 SFT/DPO 학습 프레임워크. YAML 설정 기반으로 학습을 실행한다
- **E4B_Model**: Google Gemma 4 E4B-it. 8.08B total params, 4.5B effective params의 edge 지향 모델
- **Evaluator**: evaluation_paper_style.py 모듈. Interactive/Non-Interactive 평가를 수행한다
- **Interactive_Evaluation**: 모델이 각 라운드에서 실제 예측하고, 예측 결과에 기반한 correct/incorrect 피드백을 받는 평가 방식
- **Quantization_Evaluator**: 학습된 Full SFT 모델을 4-bit/8-bit로 양자화한 후 성능을 측정하는 모듈
- **Train_Data**: 원 논문의 학습 데이터. bayesian.jsonl (31,200 records) 및 oracle.jsonl (31,200 records). 4-feature flight domain
- **Heldout_Data**: 평가용 100개 항공편 세트와 사용자별 정답 인덱스
- **Round_Accuracy**: 특정 라운드에서 Heldout_Data 100 sets에 대한 정답률
- **Learning_Effect**: R5(Round 5 정확도) - R1(Round 1 정확도)로 측정되는 학습 개선폭
- **SLURM_Job**: SLURM 클러스터에서 실행되는 학습/평가 작업 단위
- **Performance_Retention**: 양자화 후 성능 / 원본(bf16) 성능 비율. 양자화가 Bayesian reasoning에 미치는 영향을 측정한다

## Requirements

### Requirement 1: 환경 구축 및 의존성 설치

**User Story:** 연구자로서, alignment-handbook + DeepSpeed + Accelerate 기반의 Full SFT 학습 환경을 SLURM 클러스터에 구축하여 E4B 모델의 전체 파라미터 학습을 수행하고 싶다.

#### Acceptance Criteria

1. THE Full_SFT_Trainer SHALL alignment-handbook, DeepSpeed, Accelerate를 기존 conda 환경(ai, Python 3.11, PyTorch 2.10, CUDA 12.8)에 설치한다
2. WHEN 환경 설치가 완료되면, THE Full_SFT_Trainer SHALL `accelerate launch --config_file zero3.yaml scripts/run_sft.py` 명령이 정상 실행됨을 검증한다
3. THE Full_SFT_Trainer SHALL DeepSpeed ZeRO-3 설정 파일(zero3.yaml)을 6× A6000 (288GB) 환경에 맞게 구성한다
4. IF alignment-handbook의 최신 버전이 Gemma 4 chat template과 호환되지 않으면, THEN THE Full_SFT_Trainer SHALL chat_template 설정을 수동으로 지정하여 호환성을 확보한다

### Requirement 2: Full SFT 학습 설정 및 실행

**User Story:** 연구자로서, 원 논문과 동일한 하이퍼파라미터로 E4B 모델의 Full SFT를 수행하여 Bayesian teaching 효과를 재현하고 싶다.

#### Acceptance Criteria

1. THE Full_SFT_Trainer SHALL 원 논문의 하이퍼파라미터를 사용한다: global batch_size=128, learning_rate=2e-6, num_epochs=1, max_seq_length=2048, warmup_ratio=0.1, lr_scheduler=cosine
2. THE Full_SFT_Trainer SHALL 6× A6000에서 global batch_size=128을 달성하기 위해 per_device_train_batch_size와 gradient_accumulation_steps를 적절히 설정한다
3. THE Full_SFT_Trainer SHALL 원 논문의 학습 데이터(bayesian.jsonl, 31,200 records)를 alignment-handbook 호환 형식으로 로딩한다
4. THE Full_SFT_Trainer SHALL 학습 중 체크포인트를 주기적으로 저장하여 SLURM 7일 시간 제한 내 재개를 가능하게 한다
5. THE Full_SFT_Trainer SHALL Bayesian 및 Oracle 두 조건 모두에 대해 학습을 수행한다
6. WHEN 학습이 완료되면, THE Full_SFT_Trainer SHALL 최종 모델을 /abr/coss36/bayesian-teaching/models/e4b-full-bayesian/ (및 e4b-full-oracle/) 디렉토리에 저장한다
7. THE Full_SFT_Trainer SHALL 학습 설정(YAML), 학습 로그(loss curve), 실행 정보를 모델 디렉토리에 함께 저장한다

### Requirement 3: 학습 데이터 전처리

**User Story:** 연구자로서, 원 논문의 JSONL 학습 데이터를 alignment-handbook이 요구하는 형식으로 변환하여 학습 파이프라인에 투입하고 싶다.

#### Acceptance Criteria

1. THE Full_SFT_Trainer SHALL bayesian.jsonl의 messages 필드를 alignment-handbook의 chat template 형식으로 변환한다
2. THE Full_SFT_Trainer SHALL 변환된 데이터가 Gemma 4의 chat template(start_of_turn/end_of_turn)과 정확히 일치함을 검증한다
3. THE Full_SFT_Trainer SHALL 원본 데이터의 31,200 records 전체를 학습에 사용한다
4. WHEN 데이터 변환 시 토큰 길이가 max_seq_length(2048)를 초과하는 샘플이 존재하면, THE Full_SFT_Trainer SHALL 해당 샘플 수를 보고하고 truncation을 적용한다

### Requirement 4: Interactive Evaluation 수행

**User Story:** 연구자로서, Full SFT로 학습된 E4B 모델의 Bayesian reasoning 성능을 원 논문 방식의 Interactive evaluation으로 측정하고 싶다.

#### Acceptance Criteria

1. THE Evaluator SHALL Full SFT 모델을 Hugging Face Transformers로 직접 로딩하여 평가한다 (Unsloth 의존성 제거)
2. THE Evaluator SHALL 4-feature 도메인에서 30명 이상의 사용자에 대해 R1~R5 Round_Accuracy를 측정한다
3. THE Evaluator SHALL Interactive_Evaluation 모드로 평가를 수행한다 (모델 실제 예측 기반 피드백)
4. THE Evaluator SHALL 2-feature, 3-feature, 5-feature 도메인에서도 동일한 평가를 수행하여 일반화 성능을 측정한다
5. WHEN 평가 결과의 파싱 실패율이 5%를 초과하면, THE Evaluator SHALL 경고 메시지를 출력하고 실패 샘플을 로깅한다
6. THE Evaluator SHALL Base(학습 전), Bayesian, Oracle 세 조건의 결과를 비교 가능한 형식으로 저장한다

### Requirement 5: 양자화 영향 분석

**User Story:** 연구자로서, Full SFT로 학습된 모델을 양자화(int8, int4)한 후 Bayesian reasoning 성능이 얼마나 유지되는지 측정하여 edge 배포 가능성을 평가하고 싶다.

#### Acceptance Criteria

1. THE Quantization_Evaluator SHALL 학습된 Full SFT 모델(bf16)을 int8 및 int4로 양자화한다
2. THE Quantization_Evaluator SHALL bf16, int8, int4 각 조건에서 동일한 Interactive_Evaluation을 수행한다
3. THE Quantization_Evaluator SHALL 각 양자화 조건의 Performance_Retention(R5 기준)을 계산한다
4. THE Quantization_Evaluator SHALL 각 양자화 조건의 GPU 메모리 사용량(VRAM)과 추론 속도(tokens/sec)를 측정한다
5. THE Quantization_Evaluator SHALL 양자화 결과를 results/ 디렉토리에 {model}_{teaching}_{quant}_{domain}.json 형식으로 저장한다

### Requirement 6: SLURM 작업 관리

**User Story:** 연구자로서, Full SFT 학습 및 평가 작업을 SLURM 클러스터에서 효율적으로 실행하고 관리하고 싶다.

#### Acceptance Criteria

1. THE SLURM_Job SHALL Full SFT 학습에 6× A6000 GPU(partition p02, nodes DIS01-06)를 요청한다
2. THE SLURM_Job SHALL 학습 작업의 wall time을 7일(168시간) 이내로 설정한다
3. THE SLURM_Job SHALL 평가 작업에 1× A6000 GPU, 메모리 64GB를 요청한다
4. THE SLURM_Job SHALL 학습 완료 후 평가가 자동 실행되도록 job dependency(--dependency=afterok)를 설정한다
5. THE SLURM_Job SHALL stdout/stderr 로그를 logs/ 디렉토리에 저장한다
6. IF 학습 중 노드 장애가 발생하면, THEN THE SLURM_Job SHALL 마지막 체크포인트에서 재개할 수 있는 resume 스크립트를 제공한다

### Requirement 7: 논문용 결과 정리 및 시각화

**User Story:** 연구자로서, 실험 결과를 4페이지 한국어 단편 논문에 포함할 출판 품질의 그래프와 표로 정리하고 싶다.

#### Acceptance Criteria

1. WHEN 시각화가 요청되면, THE Evaluator SHALL 라운드별 정확도 변화 그래프(R1~R5)를 Base/Bayesian/Oracle 조건별로 생성한다
2. THE Evaluator SHALL 양자화 조건별(bf16/int8/int4) 성능 비교 그래프를 생성한다
3. THE Evaluator SHALL 도메인 복잡도별(2/3/4/5-feature) 일반화 성능 그래프를 생성한다
4. THE Evaluator SHALL 주요 지표(R1, R5, Learning_Effect, Performance_Retention)를 요약하는 LaTeX 표를 생성한다
5. THE Evaluator SHALL 모든 그래프를 300 DPI 이상의 해상도로 PDF 및 PNG 형식으로 저장한다
6. THE Evaluator SHALL 원 논문(Gemma 2 9B) 결과와의 비교 표를 포함한다

### Requirement 8: 추론 모듈 리팩토링

**User Story:** 연구자로서, Unsloth 의존성을 제거하고 Hugging Face Transformers 기반의 추론 모듈을 구현하여 Full SFT 모델과 양자화 모델을 통합적으로 평가하고 싶다.

#### Acceptance Criteria

1. THE Evaluator SHALL Hugging Face Transformers의 AutoModelForCausalLM으로 Full SFT 모델을 로딩한다
2. THE Evaluator SHALL bitsandbytes 라이브러리를 사용하여 int8 및 int4 양자화 로딩을 지원한다
3. THE Evaluator SHALL 기존 GemmaInference 클래스와 동일한 인터페이스(generate, generate_batch)를 유지한다
4. THE Evaluator SHALL Gemma 4의 chat template을 tokenizer.apply_chat_template()으로 적용한다
5. IF 모델 로딩 시 GPU 메모리가 부족하면, THEN THE Evaluator SHALL device_map="auto"를 사용하여 자동 분산 로딩을 수행한다
