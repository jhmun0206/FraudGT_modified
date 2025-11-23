# FraudGT 논문 재현 가이드

이 문서는 FraudGT 프로젝트의 논문 기반 실험 재현을 위한 단계별 가이드입니다.

## 📋 목차

1. [환경 설정](#1-환경-설정)
2. [데이터셋 준비](#2-데이터셋-준비)
3. [프로젝트 구조 확인](#3-프로젝트-구조-확인)
4. [설정 파일 수정](#4-설정-파일-수정)
5. [실험 실행](#5-실험-실행)
6. [문제 해결](#6-문제-해결)

---

## 1. 환경 설정

### 1.1 서버 접속 및 작업 디렉토리 확인

```bash
# 서버 접속 (이미 수행함)
ssh -p 30080 jhmun0206@moana.khu.ac.kr

# GPU 할당된 노드 접속
# 사용 가능한 파티션 확인: sinfo 또는 sinfo -o "%P %G"
srun --partition=debug_ce_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=24G --pty bash

# r4 GPU 파티션 사용 (파티션 이름은 시스템에 따라 다를 수 있음)
# 예시 1: r4 파티션이 있다면
# srun --partition=r4 --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=24G --pty bash

# 예시 2: r4 GPU 타입 지정
# srun --partition=debug_ce_ugrad --gres=gpu:r4:1 --cpus-per-gpu=8 --mem-per-gpu=24G --pty bash

# 프로젝트 디렉토리로 이동
cd /data/jhmun0206/repos/FraudGT
```

### 1.2 Conda 환경 확인 및 활성화

**중요**: `srun`으로 새 세션을 시작한 경우 conda 초기화가 필요합니다.

```bash
# Conda 초기화 (처음 한 번만 실행, 필요시)
# conda init bash

# Conda 초기화 스크립트 로드 (새 세션마다 실행)
source ~/.bashrc
# 또는 직접 conda 스크립트 로드
# source /data/jhmun0206/miniconda3/etc/profile.d/conda.sh

# Conda 환경 확인
conda env list

# fraudgt 환경 활성화
conda activate fraudgt

# 필요한 패키지 확인
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)"
```

**문제 해결**: `conda activate`가 작동하지 않으면:
```bash
# 방법 1: bashrc 다시 로드
source ~/.bashrc
conda activate fraudgt

# 방법 2: 직접 conda 스크립트 로드
source /data/jhmun0206/miniconda3/etc/profile.d/conda.sh
conda activate fraudgt

# 방법 3: conda init이 안 되어 있다면
conda init bash
source ~/.bashrc
conda activate fraudgt
```

### 1.3 필요한 패키지 설치 확인

```bash
# requirements.txt 확인
cat requirements.txt

# 필요한 경우 설치
pip install -r requirements.txt
```

---

## 2. 데이터셋 준비

### 2.1 EllipticPP 데이터셋 구조 확인

EllipticPP 데이터셋은 다음 파일들이 필요합니다:

```
/local_datasets/ellipticpp/
├── txs_features.csv   # 노드 피처 (필수)
├── txs_classes.csv    # 노드 레이블 (필수)
└── txs_edgelist.csv   # 엣지 리스트 (필수)
```

### 2.2 데이터셋 파일 확인

```bash
# 데이터셋 디렉토리 확인
ls -lh /local_datasets/ellipticpp/

# 파일 존재 여부 확인
[ -f /local_datasets/ellipticpp/txs_features.csv ] && echo "✓ features exists"
[ -f /local_datasets/ellipticpp/txs_classes.csv ] && echo "✓ classes exists"
[ -f /local_datasets/ellipticpp/txs_edgelist.csv ] && echo "✓ edges exists"
```

### 2.3 데이터셋 로더 확인

현재 `fraudGT/datasets/ellipticpp_pyg.py` 파일이 생성되어 있으며, 다음 기능을 포함합니다:

- ✅ NaN 값 처리 (fillna + 정규화 후 NaN 처리)
- ✅ 레이블 변환 (1-based → 0-based)
- ✅ PyTorch Geometric InMemoryDataset 구현

---

## 3. 프로젝트 구조 확인

### 3.1 주요 파일 및 디렉토리

```
FraudGT/
├── fraudGT/
│   ├── main.py                    # 메인 실행 파일
│   ├── datasets/
│   │   └── ellipticpp_pyg.py     # EllipticPP 데이터셋 로더 (생성됨)
│   └── loader/
│       └── master_loader.py      # 데이터셋 로더 마스터 (ellipticpp 지원)
├── configs/
│   └── ELLIPTICPP-GCN.yaml        # EllipticPP 실험 설정 파일
└── run/
    └── interactive_run.sh        # 실행 스크립트 예제
```

### 3.2 데이터셋 로더 통합 확인

`fraudGT/loader/master_loader.py`의 163-164번째 줄에서 EllipticPP 데이터셋을 자동으로 로드합니다:

```python
if name.lower() == 'ellipticpp' and str(format).lower() in ['pyg', 'pytorch_geometric', 'tg']:
    return EllipticPPPyG(dataset_dir)
```

---

## 4. 설정 파일 수정

### 4.1 ELLIPTICPP-GCN.yaml 확인 및 수정

현재 설정 파일 위치: `configs/ELLIPTICPP-GCN.yaml`

**주요 설정 항목:**

```yaml
out_dir: /data/jhmun0206/results/fraudgt/ellipticpp_gcn  # 결과 저장 경로

dataset:
  name: ellipticpp
  dir: /local_datasets/ellipticpp  # 데이터셋 경로
  format: pyg
  task: node
  task_type: classification
  split: [0.8, 0.1, 0.1]  # train/val/test 비율
  split_mode: standard
  to_undirected: True

model:
  type: gnn

gnn:
  layer_type: generalconv
  layers_pre_mp: 0
  layers_mp: 3
  layers_post_mp: 1
  dim_inner: 128
  batch_norm: True
  residual: True
  dropout: 0.2

train:
  mode: custom
  batch_size: 2048
  sampler: full_batch
  eval_period: 1

optim:
  optimizer: adam
  base_lr: 0.001
  weight_decay: 0.0001
  max_epoch: 50
  scheduler: cos

device: cuda
```

### 4.2 경로 확인

설정 파일에서 다음 경로들이 올바른지 확인:

1. **데이터셋 경로**: `dataset.dir` → `/local_datasets/ellipticpp`
2. **결과 저장 경로**: `out_dir` → `/data/jhmun0206/results/fraudgt/ellipticpp_gcn`

---

## 5. 실험 실행 전 검증

### 5.1 데이터셋 테스트 (강력 권장)

실험을 실행하기 전에 데이터셋이 올바르게 로드되는지 확인하세요:

**중요**: `test_dataset.py` 파일이 서버에 있어야 합니다. 로컬에서 생성한 경우 SFTP로 업로드해야 합니다.

```bash
# 데이터셋 테스트 스크립트 실행
python test_dataset.py
```

**파일이 없는 경우 업로드 방법**:
- VS Code SFTP 확장 사용: 파일 저장 시 자동 업로드 (uploadOnSave: true)
- 또는 수동 업로드: scp, sftp 등 사용

이 스크립트는 다음을 확인합니다:
- ✅ 필수 파일 존재 여부
- ✅ 데이터셋 로딩 성공 여부
- ✅ NaN/Inf 값 존재 여부
- ✅ 레이블 범위 및 클래스 수
- ✅ 피처 통계

**출력 예시**:
```
============================================================
EllipticPP 데이터셋 테스트
============================================================

1. 데이터셋 경로 확인: /local_datasets/ellipticpp
   ✓ txs_features.csv: True
   ✓ txs_classes.csv: True
   ✓ txs_edgelist.csv: True

2. 데이터셋 로딩 중...

3. 데이터셋 정보:
   - 노드 수: 203,769
   - 엣지 수: 234,355
   - 피처 차원: 166
   - 레이블 수: 203,769

4. 데이터 품질 검사:
   - x에 NaN: False ✓
   - x에 Inf: False ✓
   - 레이블 범위: 0 ~ 2
   - 고유 레이블: [0, 1, 2]
   - 클래스 수: 3
   - 레이블 인덱싱: ✓ (0-based)

============================================================
✅ 모든 검사를 통과했습니다! 실험을 진행할 수 있습니다.
============================================================
```

### 5.2 실험 실행

## 6. 실험 실행

### 6.1 기본 실행 명령어

```bash
# 프로젝트 루트에서 실행
cd /data/jhmun0206/repos/FraudGT

# Conda 환경 활성화
conda activate fraudgt

# 기본 실행
python -m fraudGT.main --cfg configs/ELLIPTICPP-GCN.yaml
```

### 5.2 GPU 지정 실행

```bash
# GPU 0번 사용
python -m fraudGT.main --cfg configs/ELLIPTICPP-GCN.yaml --gpu 0

# GPU 1번 사용
python -m fraudGT.main --cfg configs/ELLIPTICPP-GCN.yaml --gpu 1
```

### 5.3 반복 실험 (여러 시드)

```bash
# 5번 반복 실행 (시드 자동 증가)
python -m fraudGT.main --cfg configs/ELLIPTICPP-GCN.yaml --repeat 5
```

### 5.4 설정 오버라이드

명령줄에서 설정을 직접 변경할 수 있습니다:

```bash
python -m fraudGT.main --cfg configs/ELLIPTICPP-GCN.yaml \
    optim.base_lr 0.0005 \
    optim.max_epoch 100 \
    gnn.dim_inner 256
```

### 5.5 실행 스크립트 예제

`run/interactive_run.sh`를 참고하여 커스텀 스크립트를 만들 수 있습니다:

```bash
#!/usr/bin/env bash

cd /data/jhmun0206/repos/FraudGT
conda activate fraudgt

python -m fraudGT.main \
    --cfg configs/ELLIPTICPP-GCN.yaml \
    --gpu 0 \
    --repeat 1
```

---

## 6. 문제 해결

### 6.1 데이터셋 로딩 문제

**문제**: `data.x`에 NaN 값이 있음

**해결**: `ellipticpp_pyg.py`에서 이미 처리됨
- `fillna(0)`로 NaN 값을 0으로 채움
- 정규화 후 발생하는 NaN도 확인하여 0으로 대체

**확인 방법**:
```python
from fraudGT.datasets.ellipticpp_pyg import EllipticPPPyG
ds = EllipticPPPyG("/local_datasets/ellipticpp")
data = ds[0]
print("x NaN?", torch.isnan(data.x).any().item())  # False여야 함
```

### 6.2 레이블 범위 문제

**문제**: `data.y`가 {1,2,3} 범위로 되어 softmax 분류 시 불일치

**해결**: `ellipticpp_pyg.py`에서 자동으로 {0,1,2}로 변환

**확인 방법**:
```python
print("y unique:", torch.unique(data.y))  # tensor([0, 1, 2])여야 함
```

### 6.3 Loss가 NaN으로 출력되는 문제

**원인 분석 체크리스트**:

1. **데이터 확인**:
   ```python
   print("x has NaN?", torch.isnan(data.x).any())
   print("x has Inf?", torch.isinf(data.x).any())
   print("x stats:", data.x.min(), data.x.max(), data.x.mean())
   ```

2. **레이블 확인**:
   ```python
   print("y unique:", torch.unique(data.y))
   print("y range:", data.y.min(), data.y.max())
   print("num_classes:", len(torch.unique(data.y)))
   ```

3. **모델 출력 확인**:
   - 모델 출력이 NaN이 되는지 확인
   - 학습률이 너무 높은지 확인
   - Gradient clipping 적용 고려

### 6.4 메모리 부족 문제

**해결 방법**:

1. 배치 크기 줄이기:
   ```yaml
   train:
     batch_size: 1024  # 2048에서 줄임
   ```

2. 모델 크기 줄이기:
   ```yaml
   gnn:
     dim_inner: 64  # 128에서 줄임
   ```

### 6.5 데이터셋 경로 문제

**확인 사항**:

1. 데이터셋 파일이 올바른 경로에 있는지 확인
2. 설정 파일의 `dataset.dir` 경로 확인
3. 파일 권한 확인

```bash
# 파일 권한 확인
ls -l /local_datasets/ellipticpp/

# 읽기 권한 확인
python -c "import pandas as pd; pd.read_csv('/local_datasets/ellipticpp/txs_features.csv', nrows=1)"
```

---

## 7. 결과 확인

### 7.1 로그 파일

실험 결과는 다음 경로에 저장됩니다:

```
/data/jhmun0206/results/fraudgt/ellipticpp_gcn/
├── run_0/          # 첫 번째 실행 (시드 0)
│   ├── config.yaml
│   ├── log.txt
│   └── ...
├── run_1/          # 두 번째 실행 (시드 1)
│   └── ...
└── agg_results.yaml  # 전체 실행 결과 집계
```

### 7.2 로그 확인

```bash
# 실시간 로그 확인
tail -f /data/jhmun0206/results/fraudgt/ellipticpp_gcn/run_0/log.txt

# 최종 결과 확인
cat /data/jhmun0206/results/fraudgt/ellipticpp_gcn/agg_results.yaml
```

---

## 8. 다음 단계

### 8.1 다른 모델 실험

`configs/` 디렉토리에 다른 모델 설정 파일이 있을 수 있습니다:

```bash
ls configs/
```

### 8.2 하이퍼파라미터 튜닝

주요 튜닝 파라미터:
- `optim.base_lr`: 학습률
- `gnn.dim_inner`: 은닉 차원
- `gnn.layers_mp`: 메시지 패싱 레이어 수
- `optim.weight_decay`: 정규화 강도

### 8.3 논문 재현 완료 체크리스트

- [ ] 데이터셋 로딩 성공 (NaN 없음)
- [ ] 학습 실행 성공 (Loss가 정상적으로 감소)
- [ ] 검증/테스트 성능 측정
- [ ] 여러 시드로 실험 반복
- [ ] 결과 집계 및 분석

---

## 9. 참고 자료

### 9.1 주요 파일

- `fraudGT/main.py`: 메인 실행 파일
- `fraudGT/datasets/ellipticpp_pyg.py`: 데이터셋 로더
- `fraudGT/loader/master_loader.py`: 데이터셋 로더 마스터
- `configs/ELLIPTICPP-GCN.yaml`: 실험 설정

### 9.2 디버깅 팁

1. **작은 데이터셋으로 먼저 테스트**
2. **verbose 모드로 상세 로그 확인**
3. **단계별로 데이터 확인**

```python
# 데이터셋 테스트 스크립트
from fraudGT.datasets.ellipticpp_pyg import EllipticPPPyG
import torch

ds = EllipticPPPyG("/local_datasets/ellipticpp")
data = ds[0]

print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.size(1)}")
print(f"Features: {data.x.shape}")
print(f"Labels: {data.y.shape}")
print(f"x NaN: {torch.isnan(data.x).any()}")
print(f"y unique: {torch.unique(data.y)}")
print(f"Num classes: {len(torch.unique(data.y))}")
```

---

## 10. 문제 발생 시

문제가 발생하면 다음 정보를 확인하세요:

1. **에러 메시지 전체**
2. **설정 파일 내용**
3. **데이터셋 파일 존재 여부 및 크기**
4. **Python/Conda 환경 버전**
5. **GPU 사용 가능 여부**

이 정보들을 함께 공유하면 더 정확한 해결책을 제시할 수 있습니다.

