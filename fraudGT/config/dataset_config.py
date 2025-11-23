from fraudGT.graphgym.register import register_config
from typing import Union
from yacs.config import CfgNode as CN

@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The entity to perform the task in an heterogeneous graph dataset
    cfg.dataset.task_entity = None
    
    # Target node type for classification (overrides task_entity if set)
    cfg.dataset.target_ntype = None

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    cfg.dataset.reverse_mp = False
    cfg.dataset.add_ports = False
    
    # Heterogeneous graph mode: none | v2_cluster | v3_metapath
    cfg.dataset.hetero_mode = 'none'
    
    # Heterogeneous graph version (deprecated, use hetero_mode instead)
    cfg.dataset.hetero_version = None
    
    # ========== 방법 A: 클러스터링 기반 이종 그래프 ==========
    cfg.dataset.cluster = CN()
    cfg.dataset.cluster.algorithm = 'kmeans'  # 'kmeans' or 'minibatch_kmeans'
    cfg.dataset.cluster.num_clusters = 500  # 클러스터 개수
    cfg.dataset.cluster.seed = None  # None이면 cfg.seed 사용
    cfg.dataset.cluster.sample_ratio_for_fit = 0.2  # 클러스터링 fit용 샘플링 비율
    cfg.dataset.cluster.max_fit_samples = 30000  # 최대 fit 샘플 수
    cfg.dataset.cluster.add_cluster_cluster_edges = True  # 클러스터 간 엣지 추가 여부
    cfg.dataset.cluster.cluster_cluster_k = 5  # 각 클러스터당 상위 k개 유사 클러스터 연결
    
    # ========== 공통 상한 설정 ==========
    cfg.dataset.max_extra_nodes = 10000  # 추가 노드 수 상한
    cfg.dataset.max_extra_edges_per_type = 100000  # 엣지 타입별 상한
    
    # ========== 기존 v2 설정 (하위 호환성) ==========
    cfg.dataset.hetero_v2_sample_ratio = 0.02
    cfg.dataset.hetero_v2_max_tx_samples = 20000
    cfg.dataset.hetero_v2_min_tx_samples = 1000
    cfg.dataset.hetero_v2_max_addresses = 100000
    cfg.dataset.hetero_v2_max_tx_to_addr_edges = 50000
    cfg.dataset.hetero_v2_max_addr_to_tx_edges = 50000
    cfg.dataset.hetero_v2_avg_outputs_per_tx = 2
    cfg.dataset.hetero_v2_avg_inputs_per_tx = 1
    
    # ========== Tx+Wallet 이종 그래프 설정 ==========
    cfg.dataset.wallet_edge_sample_ratio = 1.0  # Wallet 엣지 샘플링 비율 (1.0 = 전체 사용)
    cfg.dataset.max_wallet_nodes = None  # 최대 wallet 노드 수 (None = 제한 없음)

    cfg.dataset.rand_split = False
