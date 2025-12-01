"""
EllipticPP 클러스터링 기반 이종 그래프 데이터셋 (방법 A)

아이디어: 트랜잭션을 클러스터링하여 각 클러스터를 'cluster' 노드로 생성
- 노드 타입: 'tx' (트랜잭션), 'cluster' (클러스터)
- 엣지 타입:
  - ('tx', 'belongs_to', 'cluster'): 각 tx가 속한 클러스터
  - ('tx', 'fwd', 'tx'): 트랜잭션 간 순방향 연결
  - ('tx', 'rev', 'tx'): 트랜잭션 간 역방향 연결
  - ('cluster', 'similar_to', 'cluster'): 클러스터 간 유사도 기반 연결 (선택)
"""

import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import HeteroData
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    from fraudGT.graphgym.config import cfg
except ImportError:
    cfg = None

try:
    from fraudGT.datasets.ellipticpp_pyg import EllipticPPPyG
except ImportError:
    EllipticPPPyG = None


class EllipticPPPyG_HeteroCluster(InMemoryDataset):
    """
    클러스터링 기반 이종 그래프 데이터셋
    
    트랜잭션을 KMeans로 클러스터링하여 각 클러스터를 별도 노드로 생성
    """
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # 기존 homo 데이터 로드
        if EllipticPPPyG is None:
            raise ImportError("EllipticPPPyG를 import할 수 없습니다.")
        base = EllipticPPPyG(root)
        data = base[0]  # PyG Data: x, edge_index, y, (train/val/test)_mask
        
        # 클러스터링 수행 및 이종 그래프 생성
        hetero = self._build_hetero_graph_with_clusters(data)
        
        self.data, self.slices = self.collate([hetero])
        
        # 파일 크기 출력
        processed_file = os.path.join(self.processed_dir, self.processed_file_names[0])
        if os.path.exists(processed_file):
            file_size = os.path.getsize(processed_file)
            file_size_mb = file_size / (1024 * 1024)
            print(f"[INFO] 저장된 파일 크기: {processed_file}")
            print(f"  - 크기: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    def _perform_clustering(self, features, num_clusters, algorithm='kmeans', seed=None):
        """
        트랜잭션 특징을 클러스터링
        
        Args:
            features: (num_txs, feature_dim) 텐서
            num_clusters: 클러스터 개수
            algorithm: 'kmeans' or 'minibatch_kmeans'
            seed: 랜덤 시드
            
        Returns:
            labels: (num_txs,) 각 tx가 속한 클러스터 ID
            cluster_centers: (num_clusters, feature_dim) 클러스터 중심
        """
        # Config에서 파라미터 가져오기
        if cfg is not None and hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'cluster'):
            sample_ratio = getattr(cfg.dataset.cluster, 'sample_ratio_for_fit', 0.2)
            max_fit_samples = getattr(cfg.dataset.cluster, 'max_fit_samples', 30000)
            cluster_seed = getattr(cfg.dataset.cluster, 'seed', None)
            if cluster_seed is None:
                cluster_seed = getattr(cfg, 'seed', 0) if cfg is not None else 0
        else:
            sample_ratio = 0.2
            max_fit_samples = 30000
            cluster_seed = seed if seed is not None else 0
        
        num_txs = features.size(0)
        features_np = features.cpu().numpy()
        
        print(f"[INFO] 클러스터링 시작:")
        print(f"  - 트랜잭션 수: {num_txs:,}개")
        print(f"  - 특징 차원: {features.size(1)}")
        print(f"  - 클러스터 수: {num_clusters}")
        print(f"  - 알고리즘: {algorithm}")
        print(f"  - 시드: {cluster_seed}")
        
        # 대용량 데이터의 경우 샘플링하여 fit
        if num_txs > max_fit_samples:
            num_samples = min(int(num_txs * sample_ratio), max_fit_samples)
            print(f"  - 샘플링: {num_txs:,}개 중 {num_samples:,}개로 fit 수행")
            np.random.seed(cluster_seed)
            sample_indices = np.random.choice(num_txs, num_samples, replace=False)
            fit_features = features_np[sample_indices]
        else:
            fit_features = features_np
        
        # 클러스터링 수행
        if algorithm == 'minibatch_kmeans':
            clusterer = MiniBatchKMeans(
                n_clusters=num_clusters,
                random_state=cluster_seed,
                n_init=10,
                batch_size=min(256, len(fit_features))
            )
        else:  # kmeans
            clusterer = KMeans(
                n_clusters=num_clusters,
                random_state=cluster_seed,
                n_init=10
            )
        
        print(f"  - 클러스터링 fit 중...")
        clusterer.fit(fit_features)
        
        # 모든 트랜잭션에 대해 클러스터 할당
        print(f"  - 모든 트랜잭션에 클러스터 할당 중...")
        labels = clusterer.predict(features_np)
        cluster_centers = clusterer.cluster_centers_
        
        print(f"  - 클러스터링 완료!")
        
        return labels, cluster_centers
    
    def _build_hetero_graph_with_clusters(self, data):
        """클러스터링 기반 이종 그래프 구성"""
        hetero = HeteroData()
        
        num_txs = data.x.size(0)
        
        # 1. 트랜잭션 노드 (기존 데이터)
        hetero['tx'].x = data.x
        hetero['tx'].y = data.y
        if hasattr(data, 'train_mask'):
            hetero['tx'].train_mask = data.train_mask
            hetero['tx'].val_mask = data.val_mask
            hetero['tx'].test_mask = data.test_mask
        
        # 2. 트랜잭션 간 엣지 (기존)
        edge_index = data.edge_index
        hetero[('tx', 'fwd', 'tx')].edge_index = edge_index.contiguous()
        hetero[('tx', 'rev', 'tx')].edge_index = edge_index.flip(0).contiguous()
        
        # 3. Config에서 클러스터링 파라미터 가져오기
        if cfg is not None and hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'cluster'):
            num_clusters = getattr(cfg.dataset.cluster, 'num_clusters', 500)
            algorithm = getattr(cfg.dataset.cluster, 'algorithm', 'kmeans')
            add_cluster_edges = getattr(cfg.dataset.cluster, 'add_cluster_cluster_edges', True)
            cluster_k = getattr(cfg.dataset.cluster, 'cluster_cluster_k', 5)
            max_extra_nodes = getattr(cfg.dataset, 'max_extra_nodes', 10000)
            max_extra_edges = getattr(cfg.dataset, 'max_extra_edges_per_type', 100000)
        else:
            num_clusters = 500
            algorithm = 'kmeans'
            add_cluster_edges = True
            cluster_k = 5
            max_extra_nodes = 10000
            max_extra_edges = 100000
        
        # 클러스터 수 상한 체크
        num_clusters = min(num_clusters, max_extra_nodes)
        
        # 4. 클러스터링 수행
        cluster_labels, cluster_centers = self._perform_clustering(
            data.x, num_clusters, algorithm
        )
        
        # 5. 클러스터 노드 생성
        num_clusters = len(cluster_centers)
        cluster_features = torch.tensor(cluster_centers, dtype=torch.float32)
        
        # 클러스터별 통계 계산
        cluster_stats = []
        for cid in range(num_clusters):
            tx_mask = cluster_labels == cid
            cluster_txs = torch.tensor(tx_mask, dtype=torch.bool)
            
            # 클러스터 내 tx feature 평균
            if tx_mask.sum() > 0:
                cluster_feat_mean = data.x[cluster_txs].mean(dim=0)
            else:
                cluster_feat_mean = cluster_features[cid]
            
            # 클러스터 내 fraud 비율
            if hasattr(data, 'y') and tx_mask.sum() > 0:
                cluster_labels_tx = data.y[cluster_txs]
                fraud_ratio = (cluster_labels_tx == 2).float().mean().item()  # 2 = fraud
            else:
                fraud_ratio = 0.0
            
            # 클러스터 크기 (log-scaled)
            cluster_size = tx_mask.sum()
            cluster_size_log = np.log1p(cluster_size).item()
            
            # 통계를 feature에 추가
            stats = torch.tensor([fraud_ratio, cluster_size_log], dtype=torch.float32)
            cluster_stats.append(torch.cat([cluster_feat_mean, stats]))
        
        cluster_features_with_stats = torch.stack(cluster_stats)
        hetero['cluster'].x = cluster_features_with_stats
        
        # 6. tx -> cluster 엣지 생성
        tx_to_cluster_edges = []
        for tx_idx, cluster_id in enumerate(cluster_labels):
            tx_to_cluster_edges.append([tx_idx, cluster_id])
        
        if len(tx_to_cluster_edges) > 0:
            hetero[('tx', 'belongs_to', 'cluster')].edge_index = torch.tensor(
                tx_to_cluster_edges, dtype=torch.long
            ).t().contiguous()
        
        # 7. cluster -> cluster 엣지 생성 (선택)
        cluster_to_cluster_edges = []
        if add_cluster_edges and num_clusters > 1:
            print(f"[INFO] 클러스터 간 유사도 계산 중...")
            # 클러스터 중심 간 코사인 유사도 계산
            cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
            similarity_matrix = cosine_similarity(cluster_centers_tensor.cpu().numpy())
            
            # 각 클러스터에 대해 상위 k개 유사 클러스터 연결
            for cid in range(num_clusters):
                similarities = similarity_matrix[cid]
                # 자기 자신 제외
                similarities[cid] = -1
                # 상위 k개 선택
                top_k_indices = np.argsort(similarities)[-cluster_k:][::-1]
                top_k_indices = top_k_indices[similarities[top_k_indices] > 0]  # 양수만
                
                for similar_cid in top_k_indices:
                    if len(cluster_to_cluster_edges) < max_extra_edges:
                        cluster_to_cluster_edges.append([cid, similar_cid])
            
            if len(cluster_to_cluster_edges) > 0:
                hetero[('cluster', 'similar_to', 'cluster')].edge_index = torch.tensor(
                    cluster_to_cluster_edges, dtype=torch.long
                ).t().contiguous()
        
        # 8. 통계 출력
        num_tx_fwd_edges = hetero[('tx', 'fwd', 'tx')].edge_index.size(1)
        num_tx_rev_edges = hetero[('tx', 'rev', 'tx')].edge_index.size(1)
        num_tx_cluster_edges = len(tx_to_cluster_edges)
        num_cluster_cluster_edges = len(cluster_to_cluster_edges)
        total_edges = num_tx_fwd_edges + num_tx_rev_edges + num_tx_cluster_edges + num_cluster_cluster_edges
        
        # 클러스터 크기 통계
        cluster_sizes = [np.sum(cluster_labels == cid) for cid in range(num_clusters)]
        
        print(f"[INFO] ========================================")
        print(f"[INFO] 클러스터링 기반 이종 그래프 생성 완료")
        print(f"[INFO] ========================================")
        print(f"[INFO] 노드 통계:")
        print(f"  - 트랜잭션 노드 (tx): {num_txs:,}개")
        print(f"  - 클러스터 노드 (cluster): {num_clusters:,}개")
        print(f"  - 전체 노드: {num_txs + num_clusters:,}개")
        print(f"[INFO] 엣지 통계:")
        print(f"  - ('tx', 'fwd', 'tx'): {num_tx_fwd_edges:,}개")
        print(f"  - ('tx', 'rev', 'tx'): {num_tx_rev_edges:,}개")
        print(f"  - ('tx', 'belongs_to', 'cluster'): {num_tx_cluster_edges:,}개")
        if num_cluster_cluster_edges > 0:
            print(f"  - ('cluster', 'similar_to', 'cluster'): {num_cluster_cluster_edges:,}개")
        print(f"  - 전체 엣지: {total_edges:,}개")
        print(f"[INFO] 클러스터 크기 통계:")
        print(f"  - 평균: {np.mean(cluster_sizes):.1f}개")
        print(f"  - 최소: {np.min(cluster_sizes)}개")
        print(f"  - 최대: {np.max(cluster_sizes)}개")
        print(f"  - 중앙값: {np.median(cluster_sizes):.1f}개")
        print(f"[INFO] 엣지 타입: {list(hetero.edge_types)}")
        print(f"[INFO] ========================================")
        
        return hetero
    
    @property
    def processed_file_names(self):
        return ['ellipticpp_hetero_cluster.pt']
    
    def process(self):
        pass  # __init__에서 이미 처리

