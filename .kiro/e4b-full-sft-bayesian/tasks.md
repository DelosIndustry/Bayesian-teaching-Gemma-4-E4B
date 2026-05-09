# Implementation Plan: E4B Full SFT Bayesian Teaching

## Overview

Full SFT 학습 파이프라인을 dependency 순서로 구현한다. 학습이 수 시간 소요되므로, 환경 설정 → 데이터 전처리 → 학습 설정 → 학습 제출을 최우선으로 처리하고, 학습이 돌아가는 동안 평가/양자화/시각화 코드를 작성한다.

Critical path: 환경 설치 → configs → train submit → (학습 중) evaluation code → quantization → visualization

## Tasks

- [x] 1. Environment setup and dependency installation
  - [x] 1.1 Create install_env.sh script
    - Install alignment-handbook, deepspeed, accelerate, trl, bitsandbytes, hypothesis into existing conda env (ai)
    - Verify torch 2.10 + CUDA 12.8 compatibility
    - Pin package versions for reproducibility
    - _Requirements: 1.1, 1.2_

  - [x] 1.2 Create project directory structure
    - Create bayesian-teaching/src/, configs/, scripts/, results/, figures/, logs/ directories
    - Create src/__init__.py
    - _Requirements: 1.1_

- [x] 2. Training configuration files
  - [x] 2.1 Create DeepSpeed ZeRO-3 config (configs/ds_zero3_config.json)
    - ZeRO stage 3 with overlap_comm, bf16 enabled
    - stage3_gather_16bit_weights_on_model_save: true for checkpoint saving
    - _Requirements: 1.3, 2.1_

  - [x] 2.2 Create Accelerate config (configs/accelerate_zero3.yaml)
    - 6 GPU processes, DEEPSPEED distributed type, bf16 mixed precision
    - Reference ds_zero3_config.json
    - _Requirements: 1.3, 2.1_

  - [x] 2.3 Create Bayesian SFT training config (configs/e4b_bayesian_sft.yaml)
    - model: google/gemma-4-E4B-it, per_device_train_batch_size=4, gradient_accumulation_steps=5
    - lr=2e-6, epochs=1, max_seq_length=2048, warmup_ratio=0.1, cosine scheduler
    - gradient_checkpointing=true, save_steps=200, save_total_limit=3
    - output_dir: /abr/coss36/bayesian-teaching/models/e4b-full-bayesian
    - _Requirements: 2.1, 2.2, 2.4, 2.6, 2.7_

  - [x] 2.4 Create Oracle SFT training config (configs/e4b_oracle_sft.yaml)
    - Same as Bayesian config but with oracle.jsonl data and output to e4b-full-oracle/
    - _Requirements: 2.5, 2.6_

- [x] 3. Data preprocessing and SFT entry script
  - [x] 3.1 Implement data_utils.py (src/data_utils.py)
    - BayesianDataLoader class: load_dataset(), validate_chat_format(), get_token_length_stats()
    - Load local JSONL as HuggingFace Dataset with 'messages' column
    - Handle role mapping (assistant → model if needed by Gemma 4 tokenizer)
    - Report truncation stats for samples exceeding max_seq_length=2048
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Create run_sft.py entry script (scripts/run_sft.py)
    - Based on alignment-handbook's run_sft.py pattern
    - Load YAML config, initialize SFTTrainer with local JSONL dataset
    - Integrate data_utils.py for dataset loading
    - Support --resume_from_checkpoint for SLURM restart
    - _Requirements: 2.3, 2.4, 1.2, 1.4_

- [x] 4. SLURM job scripts and training submission
  - [x] 4.1 Create training SLURM script (scripts/train.sbatch)
    - Partition p02, 6 GPUs, 48 CPUs, 256GB RAM, 168h wall time
    - Activate conda env, run accelerate launch with zero3 config
    - Log to logs/ directory
    - _Requirements: 6.1, 6.2, 6.5_

  - [x] 4.2 Create evaluation SLURM script (scripts/eval.sbatch)
    - 1 GPU, 64GB RAM, dependency on training job (--dependency=afterok)
    - Run evaluation across all domains
    - _Requirements: 6.3, 6.4, 6.5_

  - [x] 4.3 Create resume script for checkpoint recovery (scripts/resume_train.sbatch)
    - Detect latest checkpoint in output_dir
    - Restart training with --resume_from_checkpoint
    - _Requirements: 6.6, 2.4_

- [ ] 5. Checkpoint - Submit training job
  - Ensure environment is installed, configs are correct, and data loads properly
  - Submit Bayesian training job to SLURM (training takes several hours)
  - Ask the user if questions arise

- [ ] 6. Inference module (src/inference.py)
  - [ ] 6.1 Implement GemmaInference class
    - __init__: Load model with AutoModelForCausalLM, support bf16/int8/int4 via bitsandbytes
    - generate(): Single conversation inference with apply_chat_template()
    - generate_batch(): Batch inference with left-padding
    - get_memory_usage(): Report GPU VRAM stats
    - device_map="auto" for OOM fallback
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 6.2 Write property tests for inference module
    - **Property 2: Chat Template Formatting Correctness**
    - **Property 4: Prediction Parsing Correctness**
    - **Validates: Requirements 3.1, 3.2, 8.4, 4.5**

- [ ] 7. Interactive evaluation module (src/evaluation.py)
  - [ ] 7.1 Implement InteractiveEvaluator class
    - Load interaction data and heldout data from eval/ directory
    - evaluate_user(): Run 5-round interactive evaluation for one user
    - evaluate_all(): Aggregate results across all users (30+ per domain)
    - build_prompt(): Construct chat messages with history and current options
    - parse_prediction(): Extract "Flight N" from model response, return None on failure
    - Track and report parse_failure_rate, warn if > 5%
    - Save results as JSON: {model}_{teaching}_{quant}_{domain}.json
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 7.2 Write property tests for evaluation module
    - **Property 3: Round Accuracy Computation**
    - **Property 4: Prediction Parsing Correctness**
    - **Validates: Requirements 4.2, 4.5**

- [ ] 8. Quantization evaluation module (src/quantize_eval.py)
  - [ ] 8.1 Implement QuantizationEvaluator class
    - evaluate_quantized(): Load model at specified quant level, run InteractiveEvaluator
    - compare_all_quants(): Run bf16, int8, int4 evaluations for a model+domain
    - Measure GPU memory usage and tokens/sec for each quant level
    - Compute performance_retention = quantized_R5 / baseline_R5
    - Save results to results/ directory
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 8.2 Write property tests for quantization metrics
    - **Property 5: Performance Retention Computation**
    - **Property 1: Batch Size Computation Invariant** (config validation)
    - **Validates: Requirements 5.3, 2.2**

- [ ] 9. Checkpoint - Verify evaluation pipeline
  - Run evaluation on base model (pre-training) for 4-feature domain to validate pipeline
  - Ensure results JSON is generated correctly
  - Ask the user if questions arise

- [ ] 10. SLURM evaluation orchestration (scripts/eval_all.sbatch)
  - [ ] 10.1 Create comprehensive evaluation script
    - Run evaluation for all conditions: {base, bayesian, oracle} × {bf16, int8, int4} × {2,3,4,5-feature}
    - Submit Oracle training job after Bayesian completes
    - Chain evaluation jobs with --dependency=afterok
    - _Requirements: 4.4, 4.6, 5.2, 6.4_

- [ ] 11. Visualization module (src/visualization.py)
  - [ ] 11.1 Implement ResultsVisualizer class
    - plot_round_accuracy(): R1-R5 curves for Base/Bayesian/Oracle (matplotlib, 300 DPI)
    - plot_quantization_comparison(): bf16/int8/int4 bar chart
    - plot_domain_generalization(): 2/3/4/5-feature performance comparison
    - generate_latex_table(): Summary table with R1, R5, Learning_Effect, Performance_Retention
    - Save all figures as PDF + PNG at 300+ DPI to figures/ directory
    - Include comparison with original paper (Gemma 2 9B) results
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 12. Final checkpoint - End-to-end validation
  - Ensure all tests pass, ask the user if questions arise
  - Verify training completed successfully
  - Run full evaluation suite on trained models
  - Generate all figures and tables

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Critical path: Tasks 1-5 must complete ASAP to start training (hours of GPU time)
- Tasks 6-8 can be written while training runs on SLURM
- Task 9 validates the evaluation pipeline on the base model before training finishes
- Property tests use the `hypothesis` library for Python
- All code is Python, targeting the existing conda env (ai, Python 3.11, PyTorch 2.10, CUDA 12.8)
