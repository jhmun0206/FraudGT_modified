#!/bin/bash
#SBATCH -J ellipticpp_txwallet_hgt
#SBATCH -p batch_ce_ugrad          # 클러스터에 맞게 수정
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=23:59:00
#SBATCH -o logs/%x-%j.out          # 현재 레포 디렉토리 기준
#SBATCH -e logs/%x-%j.err

set -euo pipefail
set -x  # 디버깅용, 실행 커맨드 로그로 남김

# 현재 스크립트가 제출된 디렉토리(= 레포 루트)로 이동
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT_DIR"

# 로그 디렉토리 보장 (레포 내부 logs/)
mkdir -p logs

# Conda 활성화 (배치에서 확실히 되도록)
# 환경에 맞게 수정 필요: 예시는 ~/.bashrc 안에 conda 초기화가 되어 있다고 가정
if command -v conda &> /dev/null; then
    # login shell에서 이미 conda init이 되어 있다면 생략 가능
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate fraudgt
else
    echo "conda 명령을 찾을 수 없습니다. conda 환경을 수동으로 활성화하세요." >&2
fi

# 환경 정보 찍기
nvidia-smi || true
python -V
which python

# wandb 꺼두기(원하면 유지)
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1  # 버퍼링 없이 즉시 출력

echo "[`date`] Start Elliptic++ Tx+Wallet HGT Training"
python -m fraudGT.main --cfg configs/ellipticpp-txwallet-hgt.yaml

echo "[`date`] Training completed."

