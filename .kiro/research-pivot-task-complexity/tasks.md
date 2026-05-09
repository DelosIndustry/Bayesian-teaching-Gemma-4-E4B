# Implementation Plan: Research Pivot — Task Complexity

## Overview

본 구현 계획은 Bayesian Teaching 연구의 핵심 문제(비대화형 평가, 보수적 LoRA alpha)를 수정하고, 2/3/4-feature 복잡도별 체계적 실험을 수행하기 위한 코드 변경과 SLURM 파이프라인을 단계별로 구현한다. 의존성 순서: 코드 변경 → 데이터 생성 → 학습 → 평가 → 분석/시각화.

## Tasks

- [x] 1. Interactive Evaluation 구현 (evaluation_paper_style.py)
  - [x] 1.1 `build_interactive_context()` 함수 구현
    - `evaluation_paper_style.py`에 `build_interactive_context(inference, interaction_user, n_learning_rounds)` 함수 추가
    - 각 학습 라운드에서 `inference.generate()`를 호출하여 모델의 실제 예측을 받고, `parse_flight_choice()`로 파싱
    - 파싱 실패 시 `np.random.randint(0, 3)`으로 대체하고 `context_parse_failures` 카운터 증가
    - 모델 예측과 사용자 실제 선택(`round_data.user_choice`)을 비교하여 `build_feedback_message()`로 correct/incorrect 피드백 생성
    - assistant 응답은 gold label이 아닌 모델의 실제 예측 결과를 사용: `f"The best option is Flight {pred + 1}."`
    - 반환값: interactive하게 구성된 messages 리스트 (모든 eval_set에서 재사용)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6_

  - [x] 1.2 `evaluate_paper_style_interactive()` 함수 구현
    - `evaluation_paper_style.py`에 `evaluate_paper_style_interactive()` 함수 추가
    - 기존 `evaluate_paper_style()`과 동일한 시그니처 + `interactive=True` 기본값
    - 각 사용자에 대해 `build_interactive_context()`를 호출하여 학습 컨텍스트 캐싱 (user당 1번)
    - 캐싱된 학습 컨텍스트 + eval_options로 평가 메시지 구성
    - 결과 JSON에 `interactive: true`, `context_parse_failures` 필드 추가
    - 파싱 실패율 > 5% 시 경고 메시지 출력
    - _Requirements: 1.1, 1.4, 1.5, 1.6, 4.2, 4.5_

  - [x] 1.3 CLI에 `--interactive` 플래그 추가
    - `evaluation_paper_style.py`의 `__main__` 블록에 `--interactive` 인자 추가
    - `--interactive` 시 `evaluate_paper_style_interactive()` 호출, 없으면 기존 `evaluate_paper_style()` 호출
    - 결과 JSON 파일명은 동일 패턴 유지 (`{model}_{teaching}_{quant}_{domain}_n{users}_s{sets}.json`)
    - _Requirements: 1.5, 4.6_

  - [ ]* 1.4 Property test: 피드백 메시지 형식 검증 (Property 1)
    - **Property 1: Feedback message format correctness**
    - `bayesian-teaching/tests/test_feedback_message.py` 생성
    - Hypothesis로 `predicted_idx`, `true_idx` ∈ {0, 1, 2} 조합에 대해 `build_feedback_message()` 출력 형식 검증
    - correct 케이스: `"Your option Flight {X} is correct."` 정확히 일치
    - incorrect 케이스: `"Your option Flight {X} is incorrect. I prefer Flight {Y}."` 정확히 일치
    - **Validates: Requirements 1.2, 1.3**

  - [ ]* 1.5 Property test: 파싱 실패 경고 임계값 (Property 6)
    - **Property 6: Parse failure warning threshold**
    - `bayesian-teaching/tests/test_parse_warning.py` 생성
    - Hypothesis로 `(parse_failures, total_inferences)` 쌍에 대해 경고 발생 조건 검증
    - `parse_failures / total_inferences > 0.05`일 때만 경고 발생
    - **Validates: Requirements 4.5**

  - [ ]* 1.6 Unit test: Interactive evaluation mock 테스트
    - `bayesian-teaching/tests/test_interactive_eval.py` 생성
    - Mock inference 객체로 `build_interactive_context()` 전체 흐름 검증
    - 모델 예측이 틀렸을 때 incorrect 피드백이 올바르게 생성되는지 확인
    - 모델 예측이 맞았을 때 correct 피드백이 올바르게 생성되는지 확인
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Checkpoint — Interactive evaluation 코드 검증
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. LoRA Alpha 수정 및 모델 디렉토리 네이밍 (train.py)
  - [x] 3.1 E2B LoRA alpha를 32로 변경하고 `--alpha_suffix` CLI 옵션 추가
    - `train.py`의 `TRAIN_CONFIGS["e2b"]["lora_alpha"]`를 16 → 32로 변경
    - `argparse`에 `--alpha_suffix` 인자 추가 (기본값: 빈 문자열)
    - 모델 저장 경로에 alpha_suffix 반영: `{model}-{domain}-{teaching}-{suffix}` (suffix가 있을 때만)
    - `train_info.json`에 `domain` 필드 추가
    - _Requirements: 2.1, 2.3, 2.4_

  - [x] 3.2 evaluation_paper_style.py에 `--alpha_suffix` 지원 추가
    - CLI에 `--alpha_suffix` 인자 추가
    - 어댑터 경로 구성 시 alpha_suffix 반영: `models/{model}-{domain}-{teaching}-{suffix}/final`
    - _Requirements: 2.4_

  - [ ]* 3.3 Property test: 모델 디렉토리 경로 유일성 (Property 2)
    - **Property 2: Model directory path uniqueness and preservation**
    - `bayesian-teaching/tests/test_directory_naming.py` 생성
    - Hypothesis로 서로 다른 (model, domain, teaching, alpha_suffix) 튜플에 대해 경로 유일성 검증
    - 비어있지 않은 alpha_suffix가 빈 alpha_suffix와 충돌하지 않는지 검증
    - **Validates: Requirements 2.4**

- [x] 4. Checkpoint — train.py 변경 검증
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. 3-Feature 데이터 생성 SLURM 스크립트
  - [x] 5.1 `gen_3feat_data_job.sh` SLURM 스크립트 생성
    - `bayesian-teaching/gen_3feat_data_job.sh` 생성
    - 기존 SLURM 패턴 따름 (partition=p02, nodelist=DIS04)
    - GPU 불필요 (CPU 작업), `--mem=32G`, `--time=01:00:00`
    - `python data_generation_nfeat.py --domain 3feat --n_users 2000` 실행
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 5.2 Property test: 3-feature 항공편 텍스트 형식 (Property 3)
    - **Property 3: N-feature flight text format consistency**
    - `bayesian-teaching/tests/test_flight_text_format.py` 생성
    - Hypothesis로 임의의 3-feature NFeatureFlight에 대해 `to_text()` 출력이 "price:", "departure time:", "duration:" 순서를 포함하는지 검증
    - **Validates: Requirements 3.2**

  - [ ]* 5.3 Property test: 사용자 타입 균등 샘플링 (Property 4)
    - **Property 4: User type uniform sampling coverage**
    - `bayesian-teaching/tests/test_user_type_sampling.py` 생성
    - Hypothesis로 n_users ≥ 270일 때 27개 사용자 타입 전체가 샘플링되는지 검증
    - **Validates: Requirements 3.3**

- [x] 6. 2-Feature 재학습 SLURM 스크립트 (alpha=32)
  - [x] 6.1 `train_2feat_bayesian_a32_job.sh` 및 `train_2feat_oracle_a32_job.sh` 생성
    - 기존 `train_2feat_bayesian_job.sh` 패턴 기반
    - `python train.py --model e2b --teaching bayesian --domain 2feat --alpha_suffix a32` 실행
    - Oracle도 동일 패턴으로 생성
    - `--gres=gpu:1`, `--mem=96G`, `--time=12:00:00`
    - _Requirements: 2.1, 2.2, 2.4_

- [x] 7. 3-Feature 학습 SLURM 스크립트
  - [x] 7.1 `train_3feat_bayesian_job.sh` 및 `train_3feat_oracle_job.sh` 생성
    - `python train.py --model e2b --teaching bayesian --domain 3feat` 실행
    - Oracle도 동일 패턴으로 생성
    - `--gres=gpu:1`, `--mem=96G`, `--time=12:00:00`
    - _Requirements: 3.1, 4.4_

- [x] 8. Interactive 평가 SLURM 스크립트 (9개 조건)
  - [x] 8.1 2-feature 평가 스크립트 3개 생성 (base/bayesian/oracle)
    - `eval_interactive_2feat_base_job.sh`, `eval_interactive_2feat_bayesian_job.sh`, `eval_interactive_2feat_oracle_job.sh`
    - `--interactive` 플래그 추가, `--domain 2feat`, `--n_users 24`, `--n_eval_sets 100`
    - bayesian/oracle은 `--alpha_suffix a32`로 alpha=32 모델 사용
    - `--gres=gpu:1`, `--mem=64G`, `--time=24:00:00`
    - _Requirements: 1.5, 4.1, 4.6_

  - [x] 8.2 3-feature 평가 스크립트 3개 생성 (base/bayesian/oracle)
    - `eval_interactive_3feat_base_job.sh`, `eval_interactive_3feat_bayesian_job.sh`, `eval_interactive_3feat_oracle_job.sh`
    - `--interactive` 플래그 추가, `--domain 3feat`, `--n_users 24`, `--n_eval_sets 100`
    - _Requirements: 4.1, 4.6_

  - [x] 8.3 4-feature 평가 스크립트 3개 생성 (base/bayesian/oracle)
    - `eval_interactive_4feat_base_job.sh`, `eval_interactive_4feat_bayesian_job.sh`, `eval_interactive_4feat_oracle_job.sh`
    - `--interactive` 플래그 추가, `--domain 4feat`, `--n_users 24`, `--n_eval_sets 100`
    - 4-feat는 기존 alpha=16 모델 사용 (alpha_suffix 없음)
    - _Requirements: 4.1, 4.6_

- [x] 9. SLURM Orchestrator 마스터 스크립트
  - [x] 9.1 `run_all_experiments.sh` 생성
    - 전체 파이프라인을 의존성 체인으로 제출하는 마스터 스크립트
    - Phase 1: 데이터 생성 (`gen_3feat_data_job.sh`)
    - Phase 2: 학습 4개 (2feat-bay-a32, 2feat-ora-a32, 3feat-bay, 3feat-ora) — 데이터 생성 완료 후
    - Phase 3: 평가 9개 — 각 학습 완료 후 (base는 즉시)
    - Phase 4: 분석/시각화 — 모든 평가 완료 후
    - `--dependency=afterok` 사용
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ]* 9.2 Property test: 결과 파일명 패턴 (Property 7)
    - **Property 7: Result filename pattern**
    - `bayesian-teaching/tests/test_result_filename.py` 생성
    - Hypothesis로 유효한 (model, teaching, quant, domain, n_users, n_eval_sets) 조합에 대해 파일명이 `{model}_{teaching}_{quant}_{domain}_n{n_users}_s{n_eval_sets}.json` 패턴과 일치하는지 검증
    - **Validates: Requirements 4.6**

- [x] 10. Checkpoint — SLURM 스크립트 검증
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. 결과 분석 모듈 (analyze_results.py 신규)
  - [x] 11.1 `analyze_results.py` 구현
    - `bayesian-teaching/src/analyze_results.py` 신규 생성
    - `results/paper_style/` 디렉토리에서 9개 조건의 결과 JSON 로드
    - 복잡도별 Learning_Effect (R5 - R1) 계산
    - Paired t-test 또는 Wilcoxon signed-rank test로 통계적 유의성 검정
    - Bayesian vs Oracle 복잡도별 성능 차이 보고
    - 2-feature에서 유의미한 Learning_Effect 없을 경우 추가 진단 항목 출력
    - 종합 분석 결과를 `results/paper_style/analysis_summary.json`으로 저장
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 11.2 Property test: 라운드 정확도 및 Learning Effect 계산 (Property 5)
    - **Property 5: Round accuracy and learning effect computation**
    - `bayesian-teaching/tests/test_accuracy_computation.py` 생성
    - Hypothesis로 임의의 per-user accuracy 리스트에 대해 `round_accuracies[r]` = 산술 평균, `learning_effect` = `round_accuracies[4] - round_accuracies[0]` 검증
    - **Validates: Requirements 4.2**

- [x] 12. 논문용 시각화 모듈 (visualize_results.py 대폭 수정)
  - [x] 12.1 복잡도별 라운드 정확도 3-panel 그래프 구현
    - `bayesian-teaching/src/visualize_results.py` 수정 또는 신규 생성
    - Figure 1: 2-feat / 3-feat / 4-feat 3개 패널, 각 패널에 Base/Bayesian/Oracle 라인
    - 300 DPI 이상, PDF + PNG 저장
    - 그래프 내부 레이블 영문, 캡션 한글
    - _Requirements: 6.1, 6.2, 6.4_

  - [x] 12.2 Learning Effect 비교 bar chart 및 LaTeX 표 구현
    - Figure 2: 복잡도별 × 교육방식별 Learning Effect bar chart
    - LaTeX 표: Complexity × Condition × R1 × R5 × ΔEffect × p-value
    - _Requirements: 6.1, 6.3_

  - [x] 12.3 사용자별 정확도 분포 box plot 구현
    - Figure 3: 복잡도별 × 교육방식별 사용자 정확도 분포 box plot
    - _Requirements: 4.3, 6.1_

- [x] 13. 분석/시각화 SLURM 스크립트
  - [x] 13.1 `analyze_and_visualize_job.sh` 생성
    - `python analyze_results.py` + `python visualize_results.py` 순차 실행
    - GPU 불필요, `--mem=16G`, `--time=01:00:00`
    - _Requirements: 5.1_

- [x] 14. Final checkpoint — 전체 파이프라인 검증
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- 이 프로젝트는 SLURM 클러스터에서 실행되므로, GPU 작업(학습/평가)은 SLURM sbatch 스크립트로 제출해야 합니다
- 4-feature는 기존 alpha=16 모델을 재사용하므로 별도 학습이 불필요합니다
- Interactive evaluation의 추가 inference 비용은 약 2% (무시 가능)
