## Elliptic++ Tx+Wallet HGT (v3) Reproducibility Repo

이 레포는 **FraudGT 위에서 Elliptic++ Tx+Wallet 이종 그래프를 학습하는 실험(v3)** 을 다른 사람이 그대로 재현할 수 있도록 정리된 버전입니다.

### 1. 환경 설정

```bash
conda create -n fraudGT python=3.10 -y
conda activate fraudGT

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

### 2. 데이터 준비

- Elliptic++ 원본 CSV 파일들을 한 디렉토리에 모은 뒤,
- `configs/ellipticpp-txwallet-hgt.yaml` 의 `dataset.dir` 를 해당 경로로 수정합니다.

```yaml
dataset:
  dir: /path/to/ellipticpp   # 여기만 자신의 환경에 맞게 수정
```

필요한 CSV 파일 이름은 `fraudGT/datasets/ellipticpp_tx_wallet_pyg.py` 의 `raw_file_names` 에 명시되어 있습니다.

### 3. 학습 실행 (sbatch 한 번으로)

동일한 클러스터에서 SLURM을 사용하는 경우, 레포 루트에서 아래처럼 실행하면 됩니다.

```bash
git clone https://github.com/jhmun0206/FraudGT_modified.git
cd FraudGT_modified

sbatch run_ellipticpp_txwallet.sh
```

- 로그는 레포 안의 `logs/` 디렉토리에 저장됩니다.
- 사용 중인 클러스터/계정에 따라 `run_ellipticpp_txwallet.sh` 안의 **파티션 이름(`-p`)** 이나 **conda 초기화 부분**은 필요시 수정할 수 있습니다.

### 4. 실험 내용 및 코드 변경점

- 이 레포는 **Elliptic++ Tx+Wallet 이종 그래프(v3)** 실험에 필요한 코드만 정리한 버전입니다.
- 자세한 내용은 아래 문서를 참고하세요.
  - `REPRODUCTION_GUIDE.md` : 실험 전체 재현 가이드
  - `HETERO_GRAPH_GUIDE.md` : 이종 그래프 구조 및 구현 설명
  - `FRAUDGT_VS_PURE_HGT_ANALYSIS.md` : FraudGT 위 HGT vs 순수 HGT 비교 분석

이 레포를 그대로 clone 한 뒤, `configs/ellipticpp-txwallet-hgt.yaml` 의 데이터 경로만 맞추고 `sbatch run_ellipticpp_txwallet.sh` 를 실행하면 동일한 실험을 재현할 수 있습니다.

