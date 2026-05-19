# Implementation Plan: Quantization and Sample Efficiency

## Overview

본 task list 는 `requirements.md` 의 8개 Requirement (Quantization Robustness × Sample Efficiency × 선택적 Cross-Domain) 와 `design.md` 의 5개 모듈 (`scripts/run_quant_efficiency.py`, `src/inference.py` 보강, `src/evaluation.py` 보강, `src/quant_analysis.py` 신규, `src/quant_visualize.py` 신규) 을 충족하기 위한 구현 작업을 정의한다.

구현 언어는 Python (conda env `sft`, `/home/sdh/miniconda3/envs/sft/bin/python`) 이며 GPU 자원은 RTX A5000 24GB × 8 (가용성 변동, 최소 1개) 을 단일 GPU 모드 (`--device-map cuda:0`) 로 사용한다.

작업 순서는 의존성 그래프에 따라 다음 단계로 진행한다:

1. **인프라 보강** — `src/inference.py` 의 메모리 측정 일관성, `src/evaluation.py` 의 latency 측정 추가, `scripts/evaluate.py` 의 atomic write.
2. **분석 모듈 신규 작성** — `src/quant_analysis.py` (순수 함수 + 디스크 wrapper) + property tests.
3. **시각화 모듈 신규 작성** — `src/quant_visualize.py` (figure / LaTeX 표 / captions) + ordering property test.
4. **Master script 작성** — `scripts/run_quant_efficiency.py` (planning, stage dispatch, manifest, skip 정책) + planning property tests.
5. **실제 평가 실행** — int8/int4 × bayesian/oracle (4 condition, 약 200분 단일 GPU).
6. **분석/시각화 실행** — `--stage analyze`, `--stage visualize`.
7. **(선택) Cross-domain** — Hotel 도메인 6 condition (+약 300분).
8. **Final 검증** — 전체 파이프라인 smoke test 및 출판 산출물 체크.

체크포인트는 인프라 보강 직후, master script 작성 직후, 평가 완료 직후에 배치한다.

평가 시간 추정 (단일 GPU 기준):
- bf16 (재사용): 0분 (이미 완료된 baseline 두 개)
- int8 평가: 약 60분/condition × 2 = 약 120분
- int4 평가: 약 40분/condition × 2 = 약 80분
- Flight 4 condition 합계: 약 200분 (~3.3시간)
- Cross-domain 6 condition 추가: 약 300분 (~5시간)

## Tasks

- [x] 1. 사전 준비: 테스트 의존성 설치 및 출력 디렉토리 스캐폴딩
  - conda env `sft` 에 `pytest`, `hypothesis` 미설치 시 `pip install pytest hypothesis` 실행 (이미 설치된 경우 no-op)
  - `results/quant_efficiency/`, `results/quant_efficiency/figures/`, `results/quant_efficiency/tables/`, `tests/`, `tests/unit/`, `tests/integration/`, `tests/data/fake_results/` 디렉토리를 (필요 시) 생성하고 빈 `__init__.py` 를 `tests/`, `tests/unit/`, `tests/integration/` 에 추가
  - `tests/conftest.py` 에 후속 fixture 들이 사용할 minimal helper (예: `make_fake_result_file(round_accuracies, num_users)` ) 의 시그니처 placeholder 만 정의 (실제 구현은 후속 task 에서)
  - _Requirements: 7.1_

- [x] 2. 인프라 보강 — 측정 및 atomic write
  - [x] 2.1 `src/inference.py::GemmaInference` 메모리 측정 일관성 보강
    - `__init__` 의 `AutoModelForCausalLM.from_pretrained` 호출 직전에 CUDA 가용 시 `torch.cuda.empty_cache()` 와 `torch.cuda.reset_peak_memory_stats()` 를 호출하도록 추가
    - `get_memory_usage()` 가 기존 `allocated_mib`, `reserved_mib` 외에 `peak_mib` (`torch.cuda.max_memory_allocated`) 를 반환하도록 확장
    - `quantization` 인자가 `{"bf16", "int8", "int4"}` 외 값이면 즉시 `ValueError("unknown quantization label: ...")` 를 raise (silent fallback 금지)
    - _Requirements: 1.1, 2.1, 2.5, 7.2_

  - [x] 2.2 `src/evaluation.py::InteractiveEvaluator` 라운드별 latency 측정 추가
    - `RoundResult` dataclass 에 `generation_seconds: float`, `response_tokens: int` 두 필드 추가
    - `evaluate_user` 의 generation 호출 주변을 `time.perf_counter()` 로 wrap 하여 `generation_seconds` 기록, 응답에 대해 `tokenizer.encode(response, add_special_tokens=False)` 길이를 `response_tokens` 로 기록 (응답이 빈 문자열이어도 wall-clock 시간은 누락 없이 기록)
    - `evaluate_all()` 이 반환하는 summary dict 에 `latency = {mean_seconds_per_round, mean_seconds_per_user, mean_tokens_per_round, per_round_seconds}` 서브 객체 추가
    - 전체 wall-clock (`elapsed_seconds_total`) 와 사용자당 평균 (`elapsed_seconds_total / num_users`) 도 summary 에 포함되도록 검토 (이미 있으면 no-op)
    - _Requirements: 2.2, 2.3_

  - [x] 2.3 `scripts/evaluate.py` 의 결과 저장을 atomic write 로 전환
    - 결과 JSON 저장 함수에서 `tmp = path.with_suffix(path.suffix + ".tmp")` 에 먼저 dump 후 `os.replace(tmp, path)` 로 교체
    - SIGINT/OOM 발생 시 partial JSON 이 디스크에 남지 않도록 보장 (skip 정책의 안전성 — Req 7.7)
    - 결과 JSON 의 top-level 키가 `model_path`, `model_slug`, `teaching`, `quantization`, `domain`, `num_rounds`, `max_users`, `timestamp`, `memory_mib`, `elapsed_seconds_total`, `summary`, `per_user` 를 모두 포함하도록 검증 (`design.md` Result_File 스키마)
    - _Requirements: 1.5, 7.7_

  - [x] 2.4* 인프라 보강 단위 테스트
    - `tests/unit/test_inference_validation.py` 에 `GemmaInference(quantization="fp8")` 가 `ValueError` 를 raise 하는지 검증 (실제 모델 로드는 monkeypatch 로 우회)
    - `tests/unit/test_evaluation_latency_fields.py` 에 `RoundResult` 와 summary dict 의 신규 필드 (`generation_seconds`, `response_tokens`, `summary["latency"]`) 가 직렬화 가능 (json.dumps round-trip) 한지 검증
    - `tests/integration/test_atomic_write.py` 에 큰 dict 를 쓰는 도중 `KeyboardInterrupt` 시뮬레이션 시 final path 에 partial JSON 이 미존재함을 검증
    - _Requirements: 1.5, 2.3, 7.7_

- [x] 3. **체크포인트 — 인프라 보강**
  - 변경된 `src/inference.py`, `src/evaluation.py`, `scripts/evaluate.py` 가 import 에러 없이 로드되는지 단일 명령으로 확인 (`python -c "from src.inference import GemmaInference; from src.evaluation import InteractiveEvaluator"`)
  - 위 단위 테스트 (`pytest tests/unit/test_inference_validation.py tests/unit/test_evaluation_latency_fields.py tests/integration/test_atomic_write.py`) 가 모두 통과
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. 분석 모듈 신규 작성 — `src/quant_analysis.py`
  - [x] 4.1 모듈 골격과 `ConditionKey` / 데이터 로딩
    - 신규 파일 `src/quant_analysis.py` 생성. `ConditionKey = tuple[str, str, str]` 타입 alias 정의 (`teaching`, `quant`, `domain`)
    - `load_results_from_dir(base_results_dir: Path, quant_results_dir: Path) -> dict[ConditionKey, dict]` 구현
      - bf16 baseline 은 `base_results_dir/{slug}_{teaching}_bf16_{domain}.json` 에서 로드
      - int8/int4 (및 hotel) 는 `quant_results_dir/{slug}_{teaching}_{quant}_{domain}.json` 에서 로드
      - 손상된 JSON 또는 필수 키 (`summary.round_accuracies`, `per_user`) 누락 시 `BrokenResultFileError` raise (silent skip 금지)
    - 파일명 patten parser `parse_result_filename(name: str) -> tuple[str, str, str, str]` 도 함께 구현 (Property 2 의 round-trip 대상)
    - _Requirements: 1.3, 3.1, 7.7_

  - [x] 4.2 Round accuracy 와 per-user correctness 매트릭스
    - `compute_round_accuracies(per_user: list[dict], num_rounds: int = 5) -> list[float]` 구현. `InteractiveEvaluator.evaluate_all` 의 집계와 동등성 유지
    - `compute_per_user_round_correctness(per_user, num_rounds=5) -> np.ndarray` 구현 (shape `(U, R)`, dtype bool)
    - _Requirements: 3.1, 3.4_

  - [x] 4.3 Sample efficiency 지표 (R-시리즈, LE, AUC, convergence_round)
    - `compute_sample_efficiency(correct_matrix) -> dict` 구현. 반환 키: `R1..R5`, `LE`, `sample_efficiency_score (R3-R1)`, `AUC`, `convergence_round_relative`, `convergence_round_absolute`, `convergence_round_mean`, `convergence_round_median`, `learning_curve_std`
    - `_convergence_relative(correct_matrix, r5_threshold, factor=0.8)` 와 `_convergence_absolute(correct_matrix, threshold=3)` 두 helper 를 모두 구현. 미도달은 `R+1` 로 right-censored
    - AUC 는 `((R1 + R5) / 2 + R2 + R3 + R4) / 4` 공식으로 계산 (또는 `numpy.trapz` 와 동치)
    - top-level `convergence_round_mean` 과 `convergence_round_median` 은 `convergence_round_relative` 값 사용
    - `learning_curve_std` 는 사용자별 cumulative running mean 의 round-wise std 의 평균
    - _Requirements: 3.2, 3.3, 3.4_

  - [x]* 4.4 Property test — AUC 정의 일치 (Property 4)
    - `tests/unit/test_quant_analysis_properties.py` 에 추가
    - **Property 4: AUC = trapezoidal**
    - **Validates: Requirements 3.2**
    - Hypothesis: `@given(rs=st.lists(st.floats(0, 1), min_size=5, max_size=5))`, `@settings(max_examples=200)`
    - 검증: `compute_AUC(rs) == ((rs[0]+rs[4])/2 + rs[1]+rs[2]+rs[3]) / 4` 그리고 `numpy.trapz(rs, x=[1..5]) / 4` 와 일치, 모든 라운드 동일한 a 면 결과 == a, 0/1 극단값 invariant, affine 선형성
    - docstring: `Feature: quantization-and-sample-efficiency, Property 4: AUC = trapezoidal`

  - [x]* 4.5 Property test — Convergence_Round 단조 도달 (Property 6)
    - `tests/unit/test_quant_analysis_properties.py` 에 추가
    - **Property 6: Convergence_Round monotone reachability**
    - **Validates: Requirements 3.3**
    - Hypothesis: `@given(correct=arrays(dtype=bool, shape=tuples(integers(1,100), just(5))), threshold=integers(1,5))`, `@settings(max_examples=200)`
    - 검증: `1 <= conv_abs <= R+1`, `(conv_abs <= R) <=> (cum_correct >= threshold)`, threshold 에 대해 monotone non-decreasing, all-true trace 의 `conv_abs(k) == k`, all-false trace 는 항상 `R+1`. `_convergence_relative` 도 threshold 에 대해 동일한 monotone 성질 검증
    - docstring: `Feature: quantization-and-sample-efficiency, Property 6: Convergence_Round monotone reachability`

  - [x] 4.6 Robustness 지표 산출
    - `compute_robustness(per_condition: dict[ConditionKey, dict]) -> dict` 구현 (Δ_R5, Δ_LE, Robustness_Gap)
    - 추가로 `compute_per_user_delta_LE(correct_bf16, correct_quant) -> np.ndarray` 구현. user-level LE = `C[u, R-1] - C[u, 0]`, per-user Δ_LE = LE_bf16 - LE_quant. mean / median / IQR 도 함께 반환
    - `align_users_across_teachings(per_user_bayesian, per_user_oracle) -> tuple[np.ndarray, np.ndarray]` helper 로 user_idx 정렬 (Req 5.1 의 paired 검정 입력으로 사용)
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x]* 4.7 Property test — Robustness arithmetic 일관성 (Property 3)
    - `tests/unit/test_quant_analysis_properties.py` 에 추가
    - **Property 3: Robustness arithmetic consistency**
    - **Validates: Requirements 4.1, 4.2, 4.3**
    - Hypothesis: 임의의 `R5_bf16, R5_int8, LE_bf16, LE_int8` 부동소수 (0..1) 입력
    - 검증: `delta_R5 == R5_bf16 - R5_int8`, `delta_LE == LE_bf16 - LE_int8`, `gap_R5 == delta_R5_oracle - delta_R5_bayesian`, 모든 condition R5 동일 시 `delta_R5 == 0` 그리고 `gap_R5 == 0`, 입력 affine 변환 시 delta 도 동일 affine
    - docstring: `Feature: quantization-and-sample-efficiency, Property 3: Robustness arithmetic`

  - [x] 4.8 통계 검정 (Wilcoxon paired, paired t-test, 정규성 체크)
    - `run_significance_tests(per_user_delta_le_paired_diff: dict[str, np.ndarray]) -> dict` 구현
      - 입력: `{"int8": diff_LE_int8, "int4": diff_LE_int4}` (각각 paired vector `oracle_per_user_delta - bayesian_per_user_delta`)
      - Wilcoxon signed-rank: `scipy.stats.wilcoxon(diff, zero_method='wilcox', alternative='two-sided')`. statistic, p_value, n_pairs (= `(diff != 0).sum()`), rank-biserial r
      - 정규성 체크: `scipy.stats.shapiro(diff)` p > 0.05 일 때만 `paired_t` 결과를 보조 지표로 추가, 그렇지 않으면 interpretation 에 정규성 위배 경고
      - Cohen's d = `mean(diff) / std(diff, ddof=1)`
      - degenerate (`n_pairs == 0` 또는 모두 0) 시 statistic/p_value=None 으로 기록하고 interpretation="degenerate"
      - p < 0.05 인 경우 stdout 에 ANSI bold 로 강조 출력
    - 반환 dict 의 각 entry 는 `{test_name, statistic, p_value, n_pairs, effect_size, interpretation}` 구조
    - _Requirements: 5.1, 5.2, 5.3, 5.5_

  - [x] 4.9 디스크 출력 wrapper 와 atomic write
    - `atomic_write_json(path, payload)` 헬퍼 (`.tmp` → `os.replace`) 구현 후 아래 wrapper 에서 사용
    - `write_sample_efficiency_summary(out, by_cond)` — `sample_efficiency_summary.json` (Req 3.5 스키마)
    - `write_robustness_summary(out, robustness)` — `robustness_summary.json` (Req 4.4 스키마)
    - `write_significance_tests(out, tests)` — `significance_tests.json` (Req 5.4 스키마)
    - `write_efficiency_table_csv(out, results)` — Req 2.4 의 (Quant × Memory × Latency) CSV
    - `write_robustness_table_csv(out, robustness)` — Req 4.4 의 (teaching, quant, R1, R5, LE, Δ_R5, Δ_LE) CSV
    - 모든 wrapper 는 atomic write 사용
    - _Requirements: 2.4, 3.5, 4.4, 5.4_

  - [x]* 4.10 단위 테스트 — hand-crafted 예시
    - `tests/unit/test_quant_analysis_basics.py` 에 추가
    - `test_compute_round_accuracies__manual_example`: hand-crafted per_user (3 users × 5 rounds) 로 known round_accuracies 일치
    - `test_compute_AUC__all_equal_returns_input` (a ∈ {0, 0.5, 1})
    - `test_robustness_drop__simple_subtraction` (R5_bf16=0.7, R5_int8=0.65 → delta_R5=0.05)
    - `test_convergence_absolute__all_correct_returns_threshold` (`[1,1,1,1,1]` + threshold=3 → 3)
    - `test_convergence_absolute__all_wrong_returns_R_plus_1` (`[0,0,0,0,0]` → 6)
    - _Requirements: 3.2, 3.3, 4.1, 4.2_

- [x] 5. 시각화 모듈 신규 작성 — `src/quant_visualize.py`
  - [x] 5.1 모듈 골격과 figure 생성 함수
    - 신규 파일 `src/quant_visualize.py` 생성
    - `make_learning_curves_figure(by_cond, domain="flight") -> Figure`: 6개 선 (Bayesian/Oracle × bf16/int8/int4), 색=teaching, linestyle=quant, x=라운드 1..5, y=Round_Accuracy
    - `make_r5_le_bar_figure(by_cond) -> Figure`: 두 패널 (R5, LE), x=quant, hue=teaching
    - `make_scatter_per_user_efficiency(bayesian_eff, oracle_eff) -> Figure`: y=x reference line 포함
    - `make_cross_domain_robustness_figure(robustness_flight, robustness_hotel) -> Figure`: 패널=teaching, hue=domain (cross-domain 옵션 시에만 호출)
    - 모든 축/범례/제목은 영문
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 8.3_

  - [x] 5.2 디스크 저장 (atomic ordering) 과 captions / LaTeX 표
    - `save_figures(figures: list[tuple[str, Figure]], out_dir: Path) -> None`: 입력 리스트 안에 None 이 하나라도 있으면 어떤 파일도 디스크에 쓰지 않음 (atomic ordering, Req 6.4). 모두 non-None 이면 `.pdf` 와 `.png` (300 dpi) 두 형식으로 일괄 savefig
    - `write_robustness_latex_table(robustness, out)` — booktabs 형식. Columns: Teaching, Quant, R1, R5, LE, Δ_R5, Δ_LE
    - `write_sample_efficiency_latex_table(by_cond, out)` — booktabs. Columns: Teaching, Quant, R3-R1, AUC, Convergence_Round
    - `write_captions_md(out, captions: dict[str, str])` — 한글 캡션, 헤더는 figure 파일명과 1:1 대응
    - 모든 출력은 atomic write (`.tmp` → `os.replace`)
    - _Requirements: 6.4, 6.5, 6.6_

  - [x]* 5.3 Property test — Figure ordering + cross-domain 가드 (Property 7)
    - `tests/unit/test_visualize.py` 에 추가
    - **Property 7: Figure ordering and cross-domain guard**
    - **Validates: Requirements 6.4, 8.4**
    - 검증 1 (atomic ordering): `save_figures([("a", fig_a), ("b", None), ("c", fig_c)], tmp_dir)` 호출 후 `tmp_dir` 에 어떤 `.pdf` / `.png` 파일도 존재하지 않음
    - 검증 2 (성공 경로): 모든 figure 가 non-None 이면 입력 리스트의 각 이름에 대해 `.pdf` 와 `.png` 가 생성됨
    - 검증 3 (cross-domain guard): `--include-cross-domain` 미지정 실행을 mock 한 시나리오에서 `cross_domain_robustness.{pdf,png}` 가 생성되지 않음 (직접 master script 호출 또는 visualize entrypoint 의 flag 분기 검증)
    - 검증 4 (captions ↔ figures 1:1): `captions.md` 의 헤더 set 이 디스크의 figure 파일 stem set 과 일치
    - docstring: `Feature: quantization-and-sample-efficiency, Property 7: Figure ordering and cross-domain guard`

- [x] 6. Master script 신규 작성 — `scripts/run_quant_efficiency.py`
  - [x] 6.1 ExperimentCondition / planning helpers (순수 함수)
    - 신규 파일 `scripts/run_quant_efficiency.py` 생성
    - `@dataclass(frozen=True) class ExperimentCondition` 정의 (`teaching`, `quantization`, `domain`, `model_path`, `model_slug`)
    - `plan_conditions(teachings, quants, domains, model_paths) -> list[ExperimentCondition]` 구현. cartesian product, 결정론적 정렬 (`domain → teaching → quant`)
    - `resolve_result_path(cond, output_dir, legacy_results_dir) -> Path`:
      - `(teaching, bf16, flight)` 이고 `legacy_results_dir/{slug}_{teaching}_bf16_flight.json` 가 존재하면 그 경로 반환
      - 그 외 모두 `output_dir/{slug}_{teaching}_{quant}_{domain}.json`
    - `format_result_filename(slug, teaching, quant, domain) -> str` 와 `parse_result_filename(name) -> tuple` (Property 2 round-trip 대상) 정의
    - `should_skip(path, force) -> bool`: `path.exists() and not force`
    - _Requirements: 1.3, 1.4, 7.1_

  - [x]* 6.2 Property test — Quantization 라벨 white-list (Property 1)
    - `tests/unit/test_planning_properties.py` 에 추가
    - **Property 1: Quantization label white-list**
    - **Validates: Requirements 1.1, 1.3, 7.1, 7.2**
    - Hypothesis: `@given(label=st.text())`, `@settings(max_examples=200)`
    - 검증: `label in {"bf16","int8","int4"}` 이면 `ExperimentCondition` 생성 / `format_result_filename` 동작, 그 외 라벨은 master script 의 인자 검증 (`validate_quantization(label)`) 이 `ValueError` raise. `GemmaInference(quantization=label)` 도 동일 (`src/inference.py` 측 검증)
    - docstring: `Feature: quantization-and-sample-efficiency, Property 1: Quantization label white-list`

  - [x]* 6.3 Property test — Path round-trip (Property 2)
    - `tests/unit/test_planning_properties.py` 에 추가
    - **Property 2: Result file path round-trip**
    - **Validates: Requirements 1.3, 8.4**
    - Hypothesis: `@given(slug=st.from_regex(r"[a-zA-Z0-9._-]{1,30}", fullmatch=True), teaching=st.sampled_from(["bayesian","oracle"]), quant=st.sampled_from(["bf16","int8","int4"]), domain=st.sampled_from(["flight","hotel"]))`, `@settings(max_examples=200)`
    - 검증: `parse_result_filename(format_result_filename(slug, teaching, quant, domain)) == (slug, teaching, quant, domain)` 그리고 `load_results_from_dir` 가 디스크에서 발견한 파일에서 추출한 ConditionKey 가 원본과 일치 (fake fixture 디렉토리 사용)
    - docstring: `Feature: quantization-and-sample-efficiency, Property 2: Path round-trip`

  - [x] 6.4 Stage dispatch 와 subprocess 호출
    - `run_evaluate_stage(plan, output_dir, seed, force, max_users) -> list[Path]`:
      - 각 condition 에 대해 `should_skip` 확인, skip 시 매니페스트 갱신·로깅 없이 완전히 건너뜀 (Req 7.7)
      - 그 외에는 `subprocess.run(["python", "scripts/evaluate.py", "--model-path", cond.model_path, "--teaching", cond.teaching, "--quantization", cond.quantization, "--domain", cond.domain, "--num-rounds", "5", "--seed", str(seed), "--output", str(target_path)] + (["--max-users", str(max_users)] if max_users else []), check=False)` 호출
      - return code != 0 시 partial 파일 정리 후 manifest 에 `status="failed", error="cuda_oom"` 등 기록 (다음 condition 으로 진행)
      - 동일 GPU 에서 weight release 보장을 위해 condition 간 subprocess 격리 유지
    - `run_analyze_stage(result_paths, output_dir) -> dict`: `src.quant_analysis` 의 `load_results_from_dir`, `compute_*`, `run_significance_tests`, `write_*` 호출. `efficiency_table.csv`, `sample_efficiency_summary.json`, `robustness_summary.json`, `robustness_table.csv`, `significance_tests.json` 모두 생성
    - `run_visualize_stage(output_dir, include_cross_domain) -> None`: `src.quant_visualize` 호출. `figures/{learning_curves, bar_R5_LE, scatter_per_user_eff}.{pdf,png}`, `figures/captions.md`, `tables/*.tex` 생성. cross-domain 일 때만 `cross_domain_robustness.{pdf,png}` 추가
    - _Requirements: 1.2, 7.1, 7.2, 7.3, 7.4, 7.7_

  - [x] 6.5 Manifest, seed, 환경 검증
    - `write_run_manifest(output_dir, plan, seed, started_at) -> Path`: `run_manifest.json` 작성. `{schema_version, timestamp, git_commit_sha, python_version, torch_version, transformers_version, bnb_version, gpu_name, gpu_count, seed, stage, include_cross_domain, force, conditions: [...]}`. 각 condition 은 `produced` 또는 `reused` 만 기록 (skipped 는 제외; Req 7.7)
    - 시작 시점에 `os.environ["PYTHONHASHSEED"] = str(seed)`, `transformers.set_seed(seed)`, `torch.manual_seed(seed)` 호출 (Req 7.5)
    - `bitsandbytes` import 시도 후 실패 시 quantization in {int8, int4} 인 condition skip 하고 stderr 에 명확한 경고 출력
    - `--include-cross-domain` 시 `data/eval/heldout/hotel.json`, `data/eval/interaction/hotel.jsonl` 존재 검사. 누락 시 즉시 `FileNotFoundError` 와 함께 종료 (placeholder 미생성, Req 8.4)
    - condition 별 `summary.parse_failure_rate > 0.05` 시 stdout warning + manifest entry `flagged=true` (Req 1.6)
    - _Requirements: 1.6, 7.5, 7.6, 7.7, 8.4, 8.5_

  - [x] 6.6 CLI argparse 와 main 진입점
    - `argparse` 로 `--stage {evaluate,analyze,visualize,all}`, `--force`, `--include-cross-domain`, `--seed (default 42)`, `--bayesian-model (default models/gemma2-9b-bayesian)`, `--oracle-model (default models/gemma2-9b-oracle)`, `--output-dir (default results/quant_efficiency)`, `--max-users N` 정의
    - `main()` 에서 stage 별 dispatch (`all` 은 evaluate → analyze → visualize 순차). `--include-cross-domain` 시 plan 에 `domain="hotel"` 6 condition 추가
    - 종료 코드: 모든 stage 가 (실패 condition 제외) 정상 완료 시 0
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x]* 6.7 통합 테스트 — Stage dispatch (mock subprocess)
    - `tests/integration/test_master_stage_dispatch.py` 추가
    - `subprocess.run` 을 monkeypatch 하여 fake (즉시 success + fake JSON write) 로 만들고 `--stage evaluate` 호출 시 정확히 4 condition (bf16 baseline 2 개 reuse) 의 subprocess 가 호출되는지 검증
    - manifest 의 `produced`/`reused` 카운트 검증
    - _Requirements: 1.4, 7.7_

  - [x]* 6.8 Property test — Skip idempotency (Property 5)
    - `tests/integration/test_skip_idempotency.py` 추가
    - **Property 5: Skip policy idempotency**
    - **Validates: Requirements 1.4, 7.7**
    - 시나리오: subprocess monkeypatch 로 fake JSON 을 생성하는 evaluate stage 를 두 번 연속 실행 (force=False)
    - 검증: 첫 호출 후 디스크 파일들의 mtime/size/contents 가 두 번째 호출 후에도 동일, `--force=True` 일 때만 mtime 갱신, manifest 의 produced 항목 수 (첫 호출=4, 두 번째 호출=0)
    - docstring: `Feature: quantization-and-sample-efficiency, Property 5: Skip policy idempotency`

- [x] 7. **체크포인트 — Master script & 분석 인프라**
  - `python scripts/run_quant_efficiency.py --stage analyze --max-users 0` 또는 mock 데이터로 dry-run 가능 여부 확인 (실제 GPU 평가 없이 분석/시각화 코드 path 가 import 에러 없이 로드되는지 검증)
  - 모든 단위 테스트 + property 테스트 + 통합 테스트가 통과: `pytest tests/ -v`
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Smoke test — 5명 사용자 평가 한 condition
  - `python scripts/run_quant_efficiency.py --stage evaluate --max-users 5 --output-dir results/quant_efficiency_smoke`
  - 결과 JSON 의 스키마 (memory_mib, latency, per_user) 가 design 의 Result_File 스키마와 일치하는지 단발성 검증
  - 종료 후 smoke 디렉토리 정리 (또는 보존하되 본 평가 디렉토리와 분리)
  - _Requirements: 1.5, 2.1, 2.3_

- [x] 9. 본 평가 실행 — Flight 도메인 4 condition (int8/int4 × bayesian/oracle)
  - 사전 검증: GPU 가용 (`nvidia-smi`), 학습된 모델 디렉토리 (`models/gemma2-9b-bayesian`, `models/gemma2-9b-oracle`) 보존, `bitsandbytes` import 가능
  - tmux session 안에서 long-running 실행 (예: `tmux new -s quant_eval`)
  - 명령: `python scripts/run_quant_efficiency.py --stage evaluate` (bf16 baseline 2 개는 자동 reuse, 신규 4 condition 만 실행)
  - 예상 소요: 약 200분 (3.3시간) — int8 ~120분 + int4 ~80분
  - 종료 후 `results/quant_efficiency/` 에 4 신규 JSON + `run_manifest.json` 존재 검증, 각 JSON 의 `summary.round_accuracies` 가 length 5, `parse_failure_rate < 0.05` (또는 flagged)
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 2.1, 2.2, 2.3, 7.5, 7.6_

- [x] 10. 분석 stage 실행
  - 명령: `python scripts/run_quant_efficiency.py --stage analyze`
  - 산출물 검증:
    - `results/quant_efficiency/sample_efficiency_summary.json` (Req 3.5 스키마)
    - `results/quant_efficiency/robustness_summary.json` (Req 4.4 스키마)
    - `results/quant_efficiency/robustness_table.csv` (Req 4.4)
    - `results/quant_efficiency/efficiency_table.csv` (Req 2.4)
    - `results/quant_efficiency/significance_tests.json` (Req 5.4)
    - p < 0.05 검정이 stdout 에 강조 출력되는지 확인
  - _Requirements: 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 11. 시각화 stage 실행
  - 명령: `python scripts/run_quant_efficiency.py --stage visualize`
  - 산출물 검증:
    - `results/quant_efficiency/figures/learning_curves.{pdf,png}` (6 lines, Req 6.1)
    - `results/quant_efficiency/figures/bar_R5_LE.{pdf,png}` (Req 6.2)
    - `results/quant_efficiency/figures/scatter_per_user_eff.{pdf,png}` (Req 6.3)
    - `results/quant_efficiency/figures/captions.md` (한글 캡션, Req 6.5)
    - `results/quant_efficiency/tables/robustness_table.tex` (booktabs, Req 6.6)
    - `results/quant_efficiency/tables/sample_efficiency_table.tex` (booktabs, Req 6.6)
    - 모든 figure 의 축/범례/제목이 영문, 캡션은 한글
    - 300 DPI 확인 (PNG metadata)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 12. **체크포인트 — Flight 도메인 결과 무결성 검증**
  - `run_manifest.json` 의 6 condition (bf16 reused + int8/int4 produced) 모두 존재
  - `flagged=true` condition 이 있는지 확인하고 (있다면 사용자 보고)
  - 테스트 재실행: `pytest tests/ -v`
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 13. (선택) Cross-Domain — Hotel 6 condition
  - 사전 검증: `data/eval/heldout/hotel.json`, `data/eval/interaction/hotel.jsonl` 존재
  - tmux 안에서 실행: `python scripts/run_quant_efficiency.py --stage evaluate --include-cross-domain`
  - 예상 소요: +약 300분 (5시간) — Hotel bf16/int8/int4 × bayesian/oracle 6 condition
  - 종료 후 `results/quant_efficiency/{slug}_{teaching}_{quant}_hotel.json` 6개 파일 검증
  - 그다음 `python scripts/run_quant_efficiency.py --stage analyze --include-cross-domain` 와 `--stage visualize --include-cross-domain` 실행 → `cross_domain_robustness.{pdf,png}` 추가 figure 와 Robustness 비교 표가 생성
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 14. **최종 검증 — 출판 산출물 체크리스트**
  - 결과 디렉토리 구조가 `design.md` 의 결과 디렉토리 구조와 일치하는지 디렉토리 listing 으로 확인
  - 학습된 모델 디렉토리 (`models/gemma2-9b-bayesian`, `models/gemma2-9b-oracle`) 가 변경되지 않고 보존되었는지 확인
  - `pytest tests/ -v` 최종 통과
  - `python -c "from src.quant_analysis import compute_sample_efficiency, compute_robustness, run_significance_tests; from src.quant_visualize import save_figures, write_robustness_latex_table"` 로 모듈 import 정상 검증
  - manifest 의 git_commit_sha, python/torch/transformers/bnb version, gpu_name 이 모두 기록되었는지 확인
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP. Test-related sub-tasks (단위 테스트, property test, 통합 테스트) 와 cross-domain (Task 13) 가 이에 해당한다.
- 핵심 MVP path (논문 1차 결과까지): Task 1 → 2 → 3 → 4.1, 4.2, 4.3, 4.6, 4.8, 4.9 → 5.1, 5.2 → 6.1, 6.4, 6.5, 6.6 → 7 → 8 → 9 → 10 → 11 → 12 → 14. 약 4-5시간 GPU 시간 + 구현 시간.
- Property tests (Task 4.4, 4.5, 4.7, 5.3, 6.2, 6.3, 6.8) 는 7개 design properties (Property 1-7) 를 1:1 로 매핑. 가능한 한 인프라 변경 직후에 추가하면 회귀를 조기 차단할 수 있다.
- 평가 stage (Task 9, 13) 는 long-running 이므로 반드시 tmux 안에서 실행할 것. 도중 중단되어도 atomic write 와 skip 정책 (Req 7.7) 덕분에 다음 호출에서 안전하게 이어 실행됨.
- bf16 baseline 두 파일 (`results/gemma2-9b-bayesian_bayesian_bf16_flight.json`, `results/gemma2-9b-oracle_oracle_bf16_flight.json`) 은 그대로 재사용되며, evaluate stage 가 절대 덮어쓰지 않는다 (Req 1.4 + Req 7.7).
- 학습된 모델 디렉토리는 평가/분석 작업 동안 read-only 로 취급. 어떤 task 도 모델 파일을 수정하거나 삭제하지 않는다.
- 디스크 사용량은 결과 JSON 합산 < 1 GiB 으로 예상. 235 GiB 가용 디스크에 충분.
