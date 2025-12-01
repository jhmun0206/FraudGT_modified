# FraudGT 프로젝트 검토 보고서

## 📋 검토 항목 요약

1. Config 스키마(yacs) 확인
2. 모델 네트워크 등록 ('hgt')
3. 데이터셋 로더(hetero 지원)
4. 학습 루프(hetero-safe)
5. HGT 모델 구현 점검

---

## 1. Config 스키마(yacs) 확인

### ✅ 확인된 항목

**파일**: `fraudGT/graphgym/config.py` (18번째 줄: `def set_cfg`)

#### Dataset 섹션

| 키 | 스키마 존재 | 위치 | 상태 |
|---|---|---|---|
| `to_undirected` | ✅ **존재** | 222번째 줄 | `cfg.dataset.to_undirected = False` |
| `reverse_mp` | ✅ **존재** | `fraudGT/config/dataset_config.py:24` | `cfg.dataset.reverse_mp = False` |
| `add_ports` | ✅ **존재** | `fraudGT/config/dataset_config.py:25` | `cfg.dataset.add_ports = False` |
| `task_entity` | ✅ **존재** | `fraudGT/config/dataset_config.py:10` | `cfg.dataset.task_entity = None` |
| `hetero` | ⚠️ **부분적** | `fraudGT/graphgym/loader.py:83` | `getattr(cfg.dataset, 'hetero', False)` - 기본값 없음 |

**결론**: 
- ✅ `to_undirected`, `reverse_mp`, `add_ports`, `task_entity`는 모두 스키마에 정의됨
- ⚠️ `hetero`는 스키마에 명시적으로 정의되지 않았지만, `getattr`로 안전하게 사용 중

#### Train 섹션

| 키 | 스키마 존재 | 위치 | 상태 |
|---|---|---|---|
| `mode` | ✅ **존재** | 239번째 줄 | `cfg.train.mode = 'standard'` |
| `sampler` | ✅ **존재** | 245번째 줄 | `cfg.train.sampler = 'full_batch'` |
| `neighbor_sizes` | ✅ **존재** | 284번째 줄 | `cfg.train.neighbor_sizes = [20, 15, 10, 5]` |

**결론**: ✅ 모든 키가 스키마에 정의되어 있음

#### Model 섹션

| 키 | 스키마 존재 | 위치 | 상태 |
|---|---|---|---|
| `type` | ✅ **존재** | 309번째 줄 | `cfg.model.type = 'gnn'` |
| `type: hgt` | ⚠️ **값 검증 필요** | - | `'hgt'`가 허용 값인지 확인 필요 |

**결론**: 
- ✅ `cfg.model.type` 키는 존재
- ⚠️ `'hgt'` 값이 실제로 등록되어 있는지 확인 필요 (2번 항목에서 확인)

#### GNN 섹션

| 키 | 스키마 존재 | 위치 | 상태 |
|---|---|---|---|
| `layer_type` | ✅ **존재** | 362번째 줄 | `cfg.gnn.layer_type = 'generalconv'` |
| `batchnorm` | ✅ **존재** | 371번째 줄 | `cfg.gnn.batchnorm = False` |
| `batch_norm` | ⚠️ **주의** | - | YAML에서 `batch_norm: True` 사용 시 `batchnorm`으로 매핑 필요 |

**결론**: 
- ✅ `layer_type`, `batchnorm` 키는 존재
- ⚠️ YAML에서 `batch_norm`을 사용하면 `batchnorm`으로 자동 변환되는지 확인 필요
- HGT 모델에서는 `layer_type`이 직접 사용되지 않을 수 있음 (모델 내부에서 HGTConv 사용)

### 🔍 추가 확인 사항

**Config 등록 함수 위치**:
```python
# fraudGT/graphgym/config.py:18
def set_cfg(cfg):
    # 기본 설정 정의
    ...
    # 커스텀 설정 등록 (463-464번째 줄)
    for func in register.config_dict.values():
        func(cfg)
```

**커스텀 설정 등록**:
- `fraudGT/config/dataset_config.py`에서 `@register_config('dataset_cfg')`로 등록됨
- `reverse_mp`, `add_ports`, `task_entity` 모두 여기서 정의됨

---

## 2. 모델 네트워크 등록 ('hgt')

### ✅ 확인된 사항

**파일**: `fraudGT/graphgym/models/hgt.py`

#### 등록 방법

**7번째 줄**: `@register.register_network('hgt')` 데코레이터로 등록됨 ✅

```python
@register.register_network('hgt')
class HGTNet(nn.Module):
    ...
```

#### Import 경로 확인

**파일**: `fraudGT/graphgym/__init__.py`

```python
from .models import *  # noqa
from .utils import *  # noqa
import fraudGT.model.hgt  # ⚠️ 경로 불일치
```

**문제점**:
- ⚠️ `import fraudGT.model.hgt` - 실제 경로는 `fraudGT/graphgym/models/hgt.py`
- 올바른 경로: `from .models.hgt import HGTNet` 또는 `from .models import hgt`

**실제 파일 위치**:
- `fraudGT/graphgym/models/hgt.py` ✅ 존재
- `fraudGT/graphgym/models/__init__.py` ✅ 존재

#### 모델 빌더 확인

**파일**: `fraudGT/graphgym/model_builder.py`

```python
# 25번째 줄
model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out, dataset=dataset)
```

**동작 방식**:
- `network_dict`에서 `cfg.model.type` 키로 모델 클래스를 가져옴
- `'hgt'`가 등록되어 있으면 `HGTNet` 클래스를 반환

### ⚠️ 잠재적 문제

1. **Import 경로 오류**:
   - `fraudGT/graphgym/__init__.py:3`에서 `import fraudGT.model.hgt` 
   - 실제 경로는 `fraudGT/graphgym/models/hgt.py`
   - 이 import가 실패하면 모듈이 로드되지 않아 등록이 안 될 수 있음

2. **등록 확인 방법**:
   ```python
   from fraudGT.graphgym.register import network_dict
   print('NETWORK KEYS:', list(network_dict.keys()))
   # 'hgt'가 포함되어 있어야 함
   ```

### ✅ 권장 사항

**`fraudGT/graphgym/models/__init__.py`에 추가**:
```python
from .hgt import HGTNet  # noqa
```

또는

**`fraudGT/graphgym/__init__.py` 수정**:
```python
# 기존: import fraudGT.model.hgt
# 수정: from .models.hgt import HGTNet  # noqa
```

---

## 3. 데이터셋 로더(hetero 지원)

### ✅ 확인된 사항

**파일**: `fraudGT/graphgym/loader.py`

#### EllipticPP 로딩 (76-90번째 줄)

```python
if name == 'ellipticpp':
    if EllipticPPPyG is None:
        raise ValueError('ellipticpp requested, but EllipticPPPyG is not available')
    dataset = EllipticPPPyG(normalized_dir)
    # Optional: wrap into trivial hetero
    if hasattr(cfg, 'dataset') and getattr(cfg.dataset, 'hetero', False):
        data0 = dataset[0]
        dataset.data = _to_trivial_hetero(
            data0, node_type=getattr(cfg.dataset, 'task_entity', 'node')
        )
        dataset.slices = None
    return dataset
```

**동작 방식**:
- ✅ `cfg.dataset.hetero` 플래그로 HeteroData 변환
- ✅ `cfg.dataset.task_entity`로 노드 타입 지정
- ✅ `_to_trivial_hetero` 함수로 Data → HeteroData 변환

#### NeighborLoader 지원 (352-377번째 줄)

```python
elif sampler == "neighbor":
    data0 = dataset[0]
    sizes = cfg.train.neighbor_sizes[:cfg.gnn.layers_mp]
    
    if isinstance(data0, HeteroData):
        task = getattr(cfg.dataset, 'task_entity', 'node')
        if task in data0.node_types and split_mask_name in data0[task]:
            input_nodes = data0[task][split_mask_name]
    
    loader_train = NeighborLoader(
        data0,
        num_neighbors=sizes,
        batch_size=batch_size,
        shuffle=shuffle,
        input_nodes=input_nodes,  # HeteroData일 때 (node_type, mask) 형태
        ...
    )
```

**동작 방식**:
- ✅ HeteroData 감지 및 처리
- ✅ `task_entity`로 타겟 노드 타입 지정
- ✅ split mask를 `input_nodes`로 전달

#### Master Loader 확인

**파일**: `fraudGT/loader/master_loader.py`

```python
# 163-164번째 줄
if name.lower() == 'ellipticpp' and str(format).lower() in ['pyg', 'pytorch_geometric', 'tg']:
    return EllipticPPPyG(dataset_dir)
```

**현재 상태**:
- ✅ EllipticPP 로딩 지원
- ⚠️ Hetero 모드 분기는 `loader.py`에서 처리됨

### ✅ 추가 확인 사항

**HeteroData 변환 함수** (`loader.py:42-56`):
```python
def _to_trivial_hetero(d: Data, node_type: str = 'node', edge_type: str = 'to'):
    hd = HeteroData()
    hd[node_type].x = d.x
    hd[node_type].y = d.y
    # masks, edges 처리
    ...
```

**결론**: ✅ Hetero 지원이 잘 구현되어 있음

---

## 4. 학습 루프(hetero-safe)

### ✅ 확인된 사항

**파일**: `fraudGT/train/custom_train.py`

#### Batch 처리 (200-202번째 줄)

```python
if isinstance(batch, Data) or isinstance(batch, HeteroData):
    batch.split = split
    batch.to(torch.device(cfg.device))
```

**동작 방식**:
- ✅ Data와 HeteroData 모두 처리
- ✅ `.to(device)` 호출로 GPU 이동

#### 모델 Forward (208번째 줄)

```python
pred, true = model(batch)
```

**주의사항**:
- ⚠️ 모델이 `(pred, true)` 튜플을 반환해야 함
- HGT 모델은 `data.out`을 설정하고 `data`를 반환하므로, head에서 처리 필요

#### CUDA 동기화 확인

**파일**: `fraudGT/timer.py`

```python
# 79번째 줄, 279번째 줄
torch.cuda.synchronize()
```

**문제점**:
- ⚠️ `torch.cuda.is_available()` 체크 없이 호출
- CPU 환경에서 에러 발생 가능

**권장 수정**:
```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

#### HeteroData 레이블 접근

**파일**: `fraudGT/head/hetero_node.py`

```python
# 29번째 줄
task = cfg.dataset.task_entity
# 31-48번째 줄
if isinstance(batch, HeteroData):
    x = batch[task].x
    y = batch[task].y
    ...
```

**동작 방식**:
- ✅ `task_entity`로 타겟 노드 타입 지정
- ✅ HeteroData에서 해당 노드 타입의 y 추출

### ⚠️ 잠재적 문제

1. **CUDA 동기화**: `torch.cuda.is_available()` 체크 필요
2. **모델 출력 형식**: HGT 모델이 `data.out`을 설정하지만, head에서 이를 읽는지 확인 필요

---

## 5. HGT 모델 구현 점검

### ✅ 확인된 사항

**파일**: `fraudGT/graphgym/models/hgt.py`

#### 입력 처리 (42-43번째 줄)

```python
def forward(self, data):
    x_dict = {ntype: self.in_proj[ntype](data[ntype].x) for ntype in self.node_types}
```

**동작 방식**:
- ✅ HeteroData의 각 노드 타입별로 입력 projection
- ✅ `self.node_types`는 `__init__`에서 `sample.metadata()`로 설정

#### 메시지 패싱 (45-47번째 줄)

```python
for conv in self.layers:
    x_dict = conv(x_dict, data.edge_index_dict)
    x_dict = {k: self.act(self.dropout(v)) for k, v in x_dict.items()}
```

**동작 방식**:
- ✅ HGTConv로 메시지 패싱
- ✅ 각 레이어 후 activation 및 dropout 적용

#### 출력 처리 (50, 56-57번째 줄)

```python
logits = self.head(x_dict['tx'])
data.out = logits
return data
```

**동작 방식**:
- ✅ 타겟 노드 타입 'tx'의 로짓만 계산
- ✅ `data.out`에 로짓 저장
- ✅ `data` 객체 반환

### ⚠️ 잠재적 문제

1. **노드 타입 하드코딩**: `'tx'`가 하드코딩되어 있음
   - 해결: `cfg.dataset.task_entity` 사용 권장

2. **출력 형식**: `data.out`을 설정하지만, head에서 이를 읽는지 확인 필요
   - `fraudGT/head/hetero_node.py`에서 `batch[task].x`를 사용하므로, `data.out`을 읽도록 수정 필요할 수 있음

3. **Config 참조**: 
   - ✅ `cfg.gnn.dim_inner` 사용
   - ✅ `cfg.gnn.layers_mp` 사용
   - ✅ `cfg.gnn.attn_heads` 사용 (hasattr 체크)
   - ✅ `cfg.gnn.dropout` 사용

### ✅ 출력 Shape 확인

**예상 출력**:
- `logits.shape = (N_target_nodes, num_classes)`
- `N_target_nodes`: 타겟 노드 타입의 노드 수
- `num_classes`: `cfg.share.dim_out`

---

## 📊 종합 검토 결과

### ✅ 정상 동작 항목

1. **Config 스키마**: 대부분의 키가 정의되어 있음
2. **모델 등록**: 데코레이터로 등록됨 (import 경로 확인 필요)
3. **데이터셋 로더**: Hetero 지원 구현됨
4. **학습 루프**: HeteroData 처리 구현됨
5. **HGT 모델**: 기본 구조는 올바름

### ⚠️ 주의 필요 항목

1. **Import 경로**: `fraudGT/graphgym/__init__.py:3`의 경로 수정 필요
2. **CUDA 동기화**: `torch.cuda.is_available()` 체크 추가 필요
3. **노드 타입**: HGT 모델에서 `'tx'` 하드코딩 → `cfg.dataset.task_entity` 사용 권장
4. **모델 출력**: `data.out` 설정 후 head에서 읽는지 확인 필요
5. **Config 키**: `hetero` 플래그를 스키마에 명시적으로 추가 권장

### 🔧 권장 수정 사항

1. **`fraudGT/graphgym/__init__.py`**:
   ```python
   # 기존: import fraudGT.model.hgt
   # 수정: from .models.hgt import HGTNet  # noqa
   ```

2. **`fraudGT/graphgym/models/hgt.py`**:
   ```python
   # 50번째 줄: logits = self.head(x_dict['tx'])
   # 수정: logits = self.head(x_dict[cfg.dataset.task_entity])
   ```

3. **`fraudGT/timer.py`**:
   ```python
   # torch.cuda.synchronize() 전에
   if torch.cuda.is_available():
       torch.cuda.synchronize()
   ```

4. **`fraudGT/config/dataset_config.py`**:
   ```python
   cfg.dataset.hetero = False  # 명시적으로 추가
   ```

---

## 🧪 빠른 검증 명령어

### 1. Config 스키마 확인
```bash
grep -n "def set_cfg" fraudGT/graphgym/config.py
grep -n "to_undirected\|reverse_mp\|add_ports\|task_entity" fraudGT/graphgym/config.py
```

### 2. 모델 등록 확인
```bash
grep -R "register_network.*hgt" fraudGT | cat
python - <<'PY'
from fraudGT.graphgym.register import network_dict
print('NETWORK KEYS:', list(network_dict.keys()))
PY
```

### 3. 데이터셋 로더 확인
```bash
grep -n "def load_pyg" fraudGT/graphgym/loader.py
grep -n "ellipticpp" -R fraudGT | cat
```

### 4. HeteroData 처리 확인
```bash
grep -n "isinstance.*HeteroData" -R fraudGT/train | cat
grep -n "synchronize()" -R fraudGT | cat
```

---

**검토 완료일**: 2025-11-04




