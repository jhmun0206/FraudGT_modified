# 실제 이종 그래프 구현 가이드

## 📋 현재 상황

**문제**: 기존 구현은 단일 노드 타입('tx')만 사용하여 사실상 동종 그래프였습니다.
- 노드 타입: `'tx'` (트랜잭션)만 존재
- 엣지 타입: `('tx', 'fwd', 'tx')`, `('tx', 'rev', 'tx')` - 단순히 방향만 다름

**해결**: 실제 이종 그래프 구조로 변경
- 노드 타입 1: `'tx'` (트랜잭션) - 레이블이 있는 타겟 노드
- 노드 타입 2: `'address'` (주소/지갑) - 트랜잭션의 입력/출력 주소
- 엣지 타입:
  - `('tx', 'to', 'address')`: 트랜잭션 → 출력 주소
  - `('address', 'from', 'tx')`: 입력 주소 → 트랜잭션
  - `('tx', 'fwd', 'tx')`: 트랜잭션 간 순방향 연결
  - `('tx', 'rev', 'tx')`: 트랜잭션 간 역방향 연결

## 🔧 수정된 파일

### 1. 새로운 데이터셋 로더
- **파일**: `fraudGT/datasets/ellipticpp_hetero_pyg_v2.py`
- **기능**: 실제 이종 그래프 구조 생성
  - 트랜잭션 노드 + 주소 노드
  - 다양한 엣지 타입

### 2. 로더 마스터 수정
- **파일**: `fraudGT/loader/master_loader.py`
- **변경**: `hetero_version: v2` 설정 시 새로운 로더 사용

### 3. HGT 모델 수정
- **파일**: `fraudGT/graphgym/models/hgt.py`
- **변경**: `cfg.dataset.task_entity`를 우선 사용하도록 수정

### 4. 새로운 설정 파일
- **파일**: `configs/hetero-multi-v2.yaml`
- **변경**: 
  - `out_dir`: 다른 디렉토리 사용 (기존 결과 보존)
  - `hetero_version: v2`: 새로운 이종 그래프 버전 사용
  - `task_entity: tx`: 트랜잭션 노드가 타겟

## 🚀 실행 방법

### 1단계: 원본 데이터 구조 확인

```bash
cd /data/jhmun0206/repos/FraudGT
python check_dataset_structure.py
```

이 스크립트는:
- 원본 데이터에 주소 정보가 있는지 확인
- 이종 그래프 구성 가능 여부 판단

### 2단계: 실제 이종 그래프로 학습

```bash
# 방법 1: 스크립트 사용
chmod +x run_hetero_v2.sh
./run_hetero_v2.sh

# 방법 2: 직접 실행
cd /data/jhmun0206/repos/FraudGT
python fraudGT/main.py --cfg configs/hetero-multi-v2.yaml --repeat 3 --gpu 0
```

### 3단계: 결과 확인

```bash
# 기존 결과 (보존됨)
ls /data/jhmun0206/results/fraudgt/ellipticpp_multi_hetero_v1/hetero-multi/

# 새로운 결과
ls /data/jhmun0206/results/fraudgt/ellipticpp_multi_hetero_v2/hetero-multi-v2/
```

## 📊 결과 디렉토리 구조

`custom_set_out_dir` 함수에 따라:
- `out_dir` (YAML) + `config 파일 이름 (stem)` + `run_id`

```
/data/jhmun0206/results/fraudgt/
├── ellipticpp_multi_hetero_v1/    # 기존 결과 (보존)
│   └── hetero-multi/               # config 파일 이름 (hetero-multi.yaml의 stem)
│       ├── 0/                      # seed 0
│       ├── 1/                      # seed 1 (있는 경우)
│       └── agg/                    # 집계 결과
│           ├── train/
│           │   ├── stats.json
│           │   └── best.json
│           └── test/
│               ├── stats.json
│               └── best.json
└── ellipticpp_multi_hetero_v2/    # 새로운 결과
    └── hetero-multi-v2/            # config 파일 이름 (hetero-multi-v2.yaml의 stem)
        ├── 0/                      # seed 0
        ├── 1/                      # seed 1
        ├── 2/                      # seed 2
        └── agg/                    # 집계 결과
            ├── train/
            │   ├── stats.json
            │   └── best.json
            └── test/
                ├── stats.json
                └── best.json
```

## ⚠️ 주의사항

### 1. 주소 정보가 없는 경우

현재 구현은 원본 데이터에 주소 정보가 없을 경우 **시뮬레이션 주소**를 생성합니다:
- 트랜잭션의 10%를 주소로 사용
- 랜덤하게 트랜잭션-주소 연결 생성

**개선 방법**:
1. 원본 데이터에서 실제 주소 정보 추출
2. 트랜잭션 특징에서 주소 관련 컬럼 확인
3. `ellipticpp_hetero_pyg_v2.py`의 `_extract_addresses_from_features()` 메서드 수정

### 2. 실제 주소 정보 사용하기

원본 데이터에 주소 정보가 있다면:

```python
# fraudGT/datasets/ellipticpp_hetero_pyg_v2.py 수정
def _extract_addresses_from_features(self, feats, address_cols):
    """실제 주소 정보 추출"""
    all_addresses = set()
    for col in address_cols:
        if col in feats.columns:
            addresses = feats[col].dropna().unique()
            all_addresses.update(addresses)
    return list(all_addresses)
```

### 3. 다른 이종성 구현 방법

원본 데이터 구조에 따라 다른 이종성 구현 가능:
- **블록 기반**: 트랜잭션을 블록 단위로 그룹화
- **시간 기반**: 시간 윈도우로 트랜잭션 그룹화
- **클러스터 기반**: 트랜잭션 클러스터링으로 메타 노드 생성

## 📈 성능 비교

학습 완료 후 기존 결과와 비교:

```bash
# 기존 결과 (동종 그래프)
cat /data/jhmun0206/results/fraudgt/ellipticpp_multi_hetero_v1/hetero-multi/agg/test/best.json

# 새로운 결과 (실제 이종 그래프)
cat /data/jhmun0206/results/fraudgt/ellipticpp_multi_hetero_v2/hetero-multi-v2/agg/test/best.json
```

## 🔍 검증 방법

이종 그래프가 제대로 생성되었는지 확인:

```python
from fraudGT.datasets.ellipticpp_hetero_pyg_v2 import EllipticPPPyG_HeteroV2

dataset = EllipticPPPyG_HeteroV2('/data/jhmun0206/datasets/ellipticpp')
data = dataset[0]

print("노드 타입:", data.node_types)
print("엣지 타입:", data.edge_types)
print("트랜잭션 노드 수:", data['tx'].num_nodes)
print("주소 노드 수:", data['address'].num_nodes)
```

## 📝 발표용 설명

**기존**: "단일 노드 타입('tx')만 사용하여 사실상 동종 그래프"

**개선**: "실제 이종 그래프 구조 반영"
- 트랜잭션 노드와 주소 노드로 구성
- 4가지 엣지 타입으로 다양한 관계 모델링
- HGT를 통한 타입별 메시지 전달

