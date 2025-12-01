import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import HeteroData
from fraudGT.datasets.ellipticpp_pyg import EllipticPPPyG  # 기존 homo 로더 재사용

class EllipticPPPyG_Hetero(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # homo 버전 먼저 로드
        base = EllipticPPPyG(root)
        data = base[0]  # PyG Data: x, edge_index, y, (train/val/test)_mask
        # HeteroData로 포장
        hetero = HeteroData()

        # 단일 노드 타입 'tx'로 두고, 엣지 타입만 fwd/rev로 분리 (가장 간단하고 안전한 이종화)
        hetero['tx'].x = data.x
        hetero['tx'].y = data.y
        if hasattr(data, 'train_mask'):
            hetero['tx'].train_mask = data.train_mask
            hetero['tx'].val_mask   = data.val_mask
            hetero['tx'].test_mask  = data.test_mask

        edge_index = data.edge_index
        hetero[('tx', 'fwd', 'tx')].edge_index = edge_index
        hetero[('tx', 'rev', 'tx')].edge_index = edge_index.flip(0)  # 역방향 타입 추가

        self.data, self.slices = self.collate([hetero])

    @property
    def processed_file_names(self):
        # InMemoryDataset 요구사항: 더미 파일명 제공
        return ['ellipticpp_hetero_v1.pt']

    def process(self):
        pass  # 상단 __init__에서 이미 메모리에 적재했으니 별도 없음