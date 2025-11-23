#!/bin/bash
#SBATCH -J ellipticpp_txwallet_hgt
#SBATCH -p batch_ce_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=23:59:00
#SBATCH -o /data/jhmun0206/repos/FraudGT/logs/%x-%j.out
#SBATCH -e /data/jhmun0206/repos/FraudGT/logs/%x-%j.err

set -euo pipefail
set -x  # 디버깅용, 실행 커맨드 로그로 남김

# 로그 디렉토리 보장
mkdir -p /data/jhmun0206/repos/FraudGT/logs

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
export PYTHONUNBUFFERED=1  # 버퍼링 없이 즉시 출력

echo "[`date`] Start Elliptic++ Tx+Wallet HGT Training"
python -m fraudGT.main --cfg configs/ellipticpp-txwallet-hgt.yaml

echo "[`date`] Training completed."

