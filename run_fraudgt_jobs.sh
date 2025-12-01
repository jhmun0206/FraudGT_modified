#!/bin/bash
#SBATCH -J fraudgt_pe_multi
#SBATCH -p debug_ce_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:59:00
#SBATCH -o /data/jhmun0206/repos/FraudGT/logs/%x-%j.out
#SBATCH -e /data/jhmun0206/repos/FraudGT/logs/%x-%j.err

set -euo pipefail
set -x  # 디버깅용, 실행 커맨드 로그로 남김

# 로그 디렉토리 보장
mkdir -p /data/jhmun0206/repos/FraudGT/logs

# (필요시) CUDA 모듈
# module load cuda/12.1

# Conda 활성화 (배치에서 확실히 되도록)
source /data/jhmun0206/miniconda3/etc/profile.d/conda.sh
conda activate fraudgt

# 환경 정보 찍기
nvidia-smi || true
python -V
which python

cd /data/jhmun0206/repos/FraudGT

# wandb 꺼두기(원하면 유지)
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

echo "[`date`] Start PE"
python -m fraudGT.main --cfg configs/ELLIPTICPP-PE-FraudGT.yaml

echo "[`date`] Start Multi"
if [ -f configs/ELLIPTICPP-Multi-FruadGT.yaml ]; then
  python -m fraudGT.main --cfg configs/ELLIPTICPP-Multi-FruadGT.yaml
else
  python -m fraudGT.main --cfg configs/ELLIPTICPP-Multi-FraudGT.yaml
fi

echo "[`date`] All done."
