#!/bin/bash
# install_env.sh — E4B Full SFT 환경 설치
# 기존 conda env (ai, Python 3.11, PyTorch 2.10, CUDA 12.8) 위에 추가 설치
#
# 사용법:
#   bash scripts/install_env.sh

set -e

echo "================================="
echo "E4B Full SFT 환경 설치"
echo "================================="

# Conda 환경 활성화
source ~/.bashrc
conda activate ai

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# alignment-handbook (TRL + SFT 학습 프레임워크)
echo "📦 Installing alignment-handbook dependencies..."
pip install --no-deps trl>=0.12.0
pip install datasets>=3.0.0 accelerate>=1.0.0

# DeepSpeed (분산 학습)
echo "📦 Installing DeepSpeed..."
pip install deepspeed>=0.15.0

# bitsandbytes (양자화 추론)
echo "📦 Installing bitsandbytes..."
pip install bitsandbytes>=0.44.0

# 평가/시각화 의존성
echo "📦 Installing evaluation dependencies..."
pip install matplotlib>=3.8.0 scipy>=1.11.0

# Property-based testing
echo "📦 Installing testing dependencies..."
pip install hypothesis>=6.90.0 pytest>=7.4.0

# 버전 확인
echo ""
echo "================================="
echo "설치 완료. 버전 확인:"
echo "================================="
python -c "
import trl; print(f'  TRL: {trl.__version__}')
import deepspeed; print(f'  DeepSpeed: {deepspeed.__version__}')
import accelerate; print(f'  Accelerate: {accelerate.__version__}')
import bitsandbytes; print(f'  bitsandbytes: {bitsandbytes.__version__}')
import datasets; print(f'  datasets: {datasets.__version__}')
import transformers; print(f'  transformers: {transformers.__version__}')
import torch; print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
"

echo ""
echo "✅ 환경 설치 완료"
