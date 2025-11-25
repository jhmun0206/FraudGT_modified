# FraudGT 위 HGT vs 순수 HGT: 차이점 분석

## 개요

본 문서는 FraudGT 프레임워크 위에 HGT 모델을 구현한 것과 순수 HGT 모델을 독립적으로 구현한 것의 차이점을 코드 기반으로 분석합니다.

---

## 1. 통합 프레임워크 vs 독립 구현

### 1.1 FraudGT의 통합 아키텍처

**코드 근거**: `fraudGT/main.py`

```python
# fraudGT/main.py (lines 34-127)
def run():
    # 1. 설정 로드 및 검증
    set_cfg(cfg)
    load_cfg(cfg, args)
    dump_cfg(cfg)  # 설정 파일 자동 저장
    
    # 2. 재현성을 위한 시드 관리
    seed_everything(cfg.seed)
    
    # 3. 통합 파이프라인
    loaders, dataset = create_loader(returnDataset=True)
    loggers = create_logger()
    model = create_model(dataset=dataset)
    optimizer = create_optimizer(...)
    scheduler = create_scheduler(...)
    
    # 4. 학습 실행
    train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)
    
    # 5. 결과 집계 (다중 시드 실행 지원)
    agg_runs(cfg.out_dir, cfg.metric_best)
```

**차이점**:
- **FraudGT**: 모든 구성요소(데이터 로더, 모델, 옵티마이저, 로거)가 통합 프레임워크 내에서 자동으로 생성되고 연결됨
- **순수 HGT**: 각 구성요소를 수동으로 생성하고 연결해야 함

---

## 2. 설정 관리 시스템 (YACS 기반)

### 2.1 선언적 설정 관리

**코드 근거**: `fraudGT/graphgym/config.py`, `configs/ellipticpp-txwallet-hgt.yaml`

```python
# fraudGT/graphgym/config.py (lines 18-89)
def set_cfg(cfg):
    # 전역 설정 객체
    cfg = CN()
    cfg.seed = 0
    cfg.device = 'auto'
    cfg.out_dir = 'results'
    cfg.tensorboard_each_run = False
    # ... 모든 설정이 중앙 집중식으로 관리
```

```yaml
# configs/ellipticpp-txwallet-hgt.yaml
dataset:
  name: ellipticpp_tx_wallet
  target_ntype: tx  # 이종 그래프 타겟 노드 타입
  wallet_edge_sample_ratio: 1.0
  
model:
  type: hgt
  loss_fun: cross_entropy
  
train:
  mode: custom
  batch_size: 1024
  sampler: neighbor
  neighbor_sizes: [20, 15, 10]
```

**차이점**:
- **FraudGT**: YAML 파일로 모든 하이퍼파라미터를 선언적으로 관리. 설정 변경 시 코드 수정 불필요
- **순수 HGT**: 하이퍼파라미터를 코드 내 하드코딩하거나 별도 설정 파일을 직접 파싱해야 함

### 2.2 동적 설정 자동 추론

**코드 근거**: `fraudGT/graphgym/loader.py`

```python
# fraudGT/graphgym/loader.py (lines 239-313)
def set_dataset_info(dataset):
    # 이종 그래프의 경우 타겟 노드 타입 기반으로 자동 설정
    if isinstance(dataset.data, HeteroData):
        task = getattr(cfg.dataset, 'target_ntype', None) or cfg.dataset.task_entity
        # 입력 차원 자동 추론
        cfg.share.dim_in[node_type] = dataset.data.x_dict[node_type].shape[1]
        # 출력 차원 자동 추론
        y = dataset.data[task].y
        cfg.share.dim_out = int(y.max()) + 1
```

**차이점**:
- **FraudGT**: 데이터셋 구조를 분석하여 `dim_in`, `dim_out` 등을 자동으로 설정
- **순수 HGT**: 데이터셋 구조를 수동으로 분석하고 모델 파라미터를 직접 설정해야 함

---

## 3. 학습 루프 및 메트릭 관리

### 3.1 통합 학습 루프

**코드 근거**: `fraudGT/train/custom_train.py`

```python
# fraudGT/train/custom_train.py (lines 70-184)
def train_epoch(cur_epoch, logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    while True:
        batch = next(iterator, None)
        if batch is None:
            break
        
        # Forward pass
        pred, true = model(batch)
        
        # Loss 계산 (통합 loss 함수)
        loss, pred_score = compute_loss(pred, true)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation 지원
        if ((it + 1) % batch_accumulation == 0):
            optimizer.step()
            optimizer.zero_grad()
        
        # 메트릭 자동 업데이트
        logger.update_stats(true=_true, pred=_pred, loss=loss.item(), ...)
    
    # 런타임 통계 자동 수집
    runtime_stats_cuda.report_stats({
        'total': 'Total',
        'data_transfer': 'Data Transfer',
        'sampling': 'Sampling + Slicing',
        'train': 'Train',
        'forward': 'Forward',
        'loss': 'Loss',
        'backward': 'Backward'
    })
```

**차이점**:
- **FraudGT**: 
  - Gradient accumulation 자동 처리
  - 런타임 통계 자동 수집 (CUDA 이벤트 기반)
  - 배치 샘플링 에러 자동 처리 (`RuntimeError` 핸들링)
- **순수 HGT**: 위 기능들을 수동으로 구현해야 함

### 3.2 종합 메트릭 계산 및 로깅

**코드 근거**: `fraudGT/graphgym/logger.py`, `fraudGT/logger.py`

```python
# fraudGT/logger.py (lines 326-376)
def write_epoch(self, cur_epoch):
    # 기본 통계
    basic_stats = self.basic()  # loss, lr, params, time_iter, gpu_memory
    
    # 태스크별 메트릭 자동 계산
    if self.task_type == 'classification_multi':
        task_stats = self.classification_multi()  # accuracy, f1, precision, recall
    
    # 커스텀 메트릭 지원
    for custom_metric in cfg.custom_metrics:
        func = register.metric_dict.get(custom_metric)
        custom_metric_score = func(self._true, self._pred, self.task_type)
        task_stats[custom_metric] = custom_metric_score
    
    # 통계 저장
    dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
    
    # TensorBoard 자동 로깅
    if cfg.tensorboard_each_run:
        dict_to_tb(stats, self.tb_writer, cur_epoch)
```

**차이점**:
- **FraudGT**: 
  - 분류/회귀 태스크별 메트릭 자동 계산
  - JSON 및 TensorBoard 자동 로깅
  - 커스텀 메트릭 등록 시스템
- **순수 HGT**: 메트릭 계산 및 로깅을 수동으로 구현해야 함

---

## 4. 체크포인트 및 재현성 관리

### 4.1 자동 체크포인트 관리

**코드 근거**: `fraudGT/graphgym/checkpoint.py`

```python
# fraudGT/graphgym/checkpoint.py (lines 38-53)
def save_ckpt(model, optimizer=None, scheduler=None, epoch=0):
    ckpt = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    torch.save(ckpt, get_ckpt_path(epoch))

def load_ckpt(model, optimizer=None, scheduler=None, epoch=-1):
    ckpt = torch.load(get_ckpt_path(epoch))
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
```

**코드 근거**: `fraudGT/main.py` (lines 80-87)

```python
# 다중 시드 실행 지원
for run_id, seed, split_index in zip(*run_loop_settings()):
    cfg.seed = seed
    cfg.run_id = run_id
    seed_everything(cfg.seed)  # PyTorch, NumPy, Python random 모두 설정
    # 각 실행마다 별도 디렉토리 생성
    custom_set_run_dir(cfg, run_id)
```

**차이점**:
- **FraudGT**: 
  - 모델, 옵티마이저, 스케줄러 상태를 통합 저장/로드
  - 다중 시드 실행 시 자동으로 별도 디렉토리 생성
  - PyTorch Geometric의 `seed_everything`으로 완전한 재현성 보장
- **순수 HGT**: 체크포인트 저장/로드 및 재현성 관리를 수동으로 구현해야 함

### 4.2 결과 집계 시스템

**코드 근거**: `fraudGT/main.py` (lines 128-131)

```python
# 다중 실행 결과 자동 집계
agg_runs(cfg.out_dir, cfg.metric_best)
```

**차이점**:
- **FraudGT**: 여러 시드/스플릿 실행 결과를 자동으로 집계하여 평균/표준편차 계산
- **순수 HGT**: 결과 집계를 수동으로 수행해야 함

---

## 5. 데이터 로딩 및 전처리 통합

### 5.1 이종 그래프 데이터셋 통합

**코드 근거**: `fraudGT/datasets/ellipticpp_tx_wallet_pyg.py`, `fraudGT/loader/master_loader.py`

```python
# fraudGT/datasets/ellipticpp_tx_wallet_pyg.py (lines 18-549)
class EllipticPPPyG_TxWallet(InMemoryDataset):
    def process(self):
        # CSV 파일 자동 로드
        txs_feats = pd.read_csv(os.path.join(raw_dir, 'txs_features.csv'))
        # ... 이종 그래프 자동 생성
        hetero = HeteroData()
        hetero['tx'].x = tx_x
        hetero['tx'].y = tx_y
        hetero['wallet'].x = wallet_x
        # 엣지 타입별 자동 생성
        hetero[('tx', 'to', 'wallet')].edge_index = edge_index_tx_to_wallet
        # ... 저장 및 로드 자동화
```

```python
# fraudGT/loader/master_loader.py (lines 195-200)
if name.lower() == 'ellipticpp_tx_wallet':
    return EllipticPPPyG_TxWallet(dataset_dir)
```

**차이점**:
- **FraudGT**: 
  - `InMemoryDataset` 상속으로 자동 캐싱
  - 데이터셋 이름만으로 자동 로드
  - 전처리 결과 자동 저장/로드
- **순수 HGT**: 데이터 로딩 및 전처리 코드를 직접 작성해야 함

### 5.2 Neighbor Sampling 통합

**코드 근거**: `fraudGT/graphgym/loader.py` (lines 355-383)

```python
# fraudGT/graphgym/loader.py
if sampler == "neighbor":
    if isinstance(data0, HeteroData):
        # 이종 그래프의 경우 타겟 노드 타입 기반 샘플링
        task = getattr(cfg.dataset, 'target_ntype', None) or getattr(cfg.dataset, 'task_entity', 'node')
        if task in data0.node_types and split_mask_name in data0[task]:
            input_nodes = (task, data0[task][split_mask_name])
    
    loader_train = NeighborLoader(
        data0,
        num_neighbors=cfg.train.neighbor_sizes,  # [20, 15, 10]
        batch_size=cfg.train.batch_size,
        input_nodes=input_nodes,  # 타겟 노드 타입만 샘플링
        num_workers=cfg.num_workers
    )
```

**차이점**:
- **FraudGT**: 
  - 이종 그래프에서 타겟 노드 타입만 샘플링하도록 자동 설정
  - Train/Val/Test split 마스크 자동 인식
- **순수 HGT**: Neighbor sampling을 수동으로 설정해야 함

---

## 6. 이종 그래프 지원 확장

### 6.1 타겟 노드 타입 기반 분류

**코드 근거**: `fraudGT/graphgym/models/hgt.py` (lines 46-110)

```python
# fraudGT/graphgym/models/hgt.py
class HGTNet(nn.Module):
    def __init__(self, dim_in, dim_out, dataset=None, **kwargs):
        # 타겟 노드 타입 우선순위 설정
        if hasattr(cfg.dataset, 'target_ntype') and cfg.dataset.target_ntype:
            self.target_ntype = cfg.dataset.target_ntype
        elif hasattr(cfg.dataset, 'task_entity') and cfg.dataset.task_entity:
            self.target_ntype = cfg.dataset.task_entity
        else:
            self.target_ntype = self.node_types[0]
    
    def forward(self, data):
        # 모든 노드 타입에 대해 임베딩 계산
        x_dict = {ntype: self.in_proj[ntype](data[ntype].x) for ntype in self.node_types}
        for conv in self.layers:
            x_dict = conv(x_dict, data.edge_index_dict)
        
        # 타겟 노드 타입만 예측
        logits = self.head(x_dict[self.target_ntype])
        true = data[self.target_ntype].y
        return logits, true
```

**코드 근거**: `fraudGT/graphgym/loader.py` (lines 243-244, 273-274)

```python
# 이종 그래프에서 타겟 노드 타입 자동 인식
task = getattr(cfg.dataset, 'target_ntype', None) or cfg.dataset.task_entity
cfg.share.dim_in[node_type] = dataset.data.x_dict[node_type].shape[1]
y = dataset.data[task].y  # 타겟 노드 타입의 레이블만 사용
```

**차이점**:
- **FraudGT**: 
  - 설정 파일에서 `target_ntype: tx` 지정만으로 tx 노드만 분류하도록 자동 설정
  - 로더, 모델, 메트릭 계산 모두 타겟 노드 타입을 자동 인식
- **순수 HGT**: 타겟 노드 타입 필터링을 각 단계에서 수동으로 구현해야 함

---

## 7. 요약: 핵심 차이점

| 항목 | FraudGT 위 HGT | 순수 HGT |
|------|---------------|----------|
| **설정 관리** | YAML 기반 선언적 설정, 자동 검증 | 코드 내 하드코딩 또는 수동 파싱 |
| **데이터 로딩** | 통합 데이터셋 클래스, 자동 캐싱 | 수동 구현 |
| **학습 루프** | 통합 학습 함수, gradient accumulation, 에러 핸들링 | 수동 구현 |
| **메트릭 계산** | 태스크별 자동 계산, 커스텀 메트릭 지원 | 수동 구현 |
| **로깅** | JSON + TensorBoard 자동 로깅 | 수동 구현 |
| **체크포인트** | 모델/옵티마이저/스케줄러 통합 저장 | 수동 구현 |
| **재현성** | 다중 시드 실행, 자동 결과 집계 | 수동 구현 |
| **이종 그래프** | 타겟 노드 타입 자동 인식 및 처리 | 수동 필터링 |
| **Neighbor Sampling** | 이종 그래프 타겟 노드 타입 자동 샘플링 | 수동 설정 |
| **런타임 통계** | CUDA 이벤트 기반 자동 수집 | 수동 구현 |

---

## 8. 결론

FraudGT 프레임워크 위에 HGT를 구현한 경우:

1. **개발 효율성**: 반복적인 보일러플레이트 코드 작성 불필요
2. **재현성**: 설정 파일 기반으로 완전한 실험 재현 가능
3. **확장성**: 새로운 데이터셋/모델 추가 시 최소한의 코드 수정
4. **유지보수성**: 중앙 집중식 설정 관리로 하이퍼파라미터 변경 용이
5. **실험 관리**: 다중 시드 실행 및 결과 집계 자동화

순수 HGT 구현의 경우 위 기능들을 모두 수동으로 구현해야 하므로 개발 시간이 크게 증가하고, 실험 재현성과 관리가 어려워집니다.

