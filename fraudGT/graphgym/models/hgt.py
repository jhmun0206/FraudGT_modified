import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from fraudGT.graphgym.config import cfg
from torch_geometric.data import HeteroData, Data

class HGTNet(nn.Module):
    """
    간단한 HGT 스택: 단일 노드 타입 'tx', 엣지 타입 ('fwd','rev') 가정
    """
    def __init__(self, dim_in, dim_out, dataset=None, **kwargs):
        super().__init__()
        assert dataset is not None, "dataset (for metadata) is required for HGT"

        # --- Determine a HeteroData sample correctly ---
        # Many loaders keep the hetero graph at `dataset.data` instead of `dataset[0]`
        sample = getattr(dataset, "data", None)
        if sample is None:
            # fallback to first element when available
            try:
                sample = dataset[0]
            except Exception:
                sample = None

        if not isinstance(sample, HeteroData):
            # Accept homogeneous Data by wrapping into a single-type HeteroData
            if isinstance(sample, Data):
                ntype = getattr(cfg.gnn, "target_ntype", "node")
                rtype = getattr(cfg.gnn, "edge_type", "rel")
                hd = HeteroData()
                if hasattr(sample, "x"):
                    hd[ntype].x = sample.x
                if hasattr(sample, "y"):
                    hd[ntype].y = sample.y
                if hasattr(sample, "edge_index"):
                    hd[(ntype, rtype, ntype)].edge_index = sample.edge_index
                sample = hd
            else:
                raise TypeError("HGTNet expects HeteroData or Data; got unknown type.")

        # Hetero metadata
        self.node_types, self.edge_types = sample.metadata()

        dim_hidden = cfg.gnn.dim_inner

        # Target node type (default to first type if not specified)
        # 우선순위: cfg.dataset.target_ntype > cfg.dataset.task_entity > cfg.gnn.target_ntype > 첫 번째 노드 타입
        if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'target_ntype') and cfg.dataset.target_ntype:
            self.target_ntype = cfg.dataset.target_ntype
        elif hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'task_entity') and cfg.dataset.task_entity:
            self.target_ntype = cfg.dataset.task_entity
        else:
            self.target_ntype = getattr(cfg.gnn, "target_ntype", self.node_types[0])
        self.single_ntype = getattr(cfg.gnn, "target_ntype", "node")
        self.single_rtype = getattr(cfg.gnn, "edge_type", "rel")

        # --- Type-specific input projections ---
        # Infer per-type input dims from sample; fall back to `dim_in` if missing.
        in_dims = {}
        for ntype in self.node_types:
            xi = getattr(sample[ntype], "x", None)
            in_dims[ntype] = xi.size(1) if xi is not None else dim_in

        self.in_proj = nn.ModuleDict({
            ntype: nn.Linear(in_dims[ntype], dim_hidden) for ntype in self.node_types
        })

        # --- HGTConv stack ---
        self.layers = nn.ModuleList([
            HGTConv(in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    metadata=(self.node_types, self.edge_types),
                    heads=getattr(cfg.gnn, 'attn_heads', 1))
            for _ in range(cfg.gnn.layers_mp)
        ])

        self.dropout = nn.Dropout(cfg.gnn.dropout)

        # Output head for target node type
        self.head = nn.Linear(dim_hidden, dim_out)
        self.act = nn.ReLU()

    def forward(self, data):
        # Wrap homogeneous Data into HeteroData at runtime if needed
        if isinstance(data, Data):
            hd = HeteroData()
            ntype = self.single_ntype
            rtype = self.single_rtype
            if hasattr(data, "x"):
                hd[ntype].x = data.x
            if hasattr(data, "y"):
                hd[ntype].y = data.y
            if hasattr(data, "edge_index"):
                hd[(ntype, rtype, ntype)].edge_index = data.edge_index
            data = hd

        x_dict = {ntype: self.in_proj[ntype](data[ntype].x) for ntype in self.node_types}

        for conv in self.layers:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: self.act(self.dropout(v)) for k, v in x_dict.items()}

        # 타겟 노드 타입만 예측 (노드 분류)
        logits = self.head(x_dict[self.target_ntype])
        # Return (pred, true) to match custom_train expectation
        if isinstance(data, Data):
            true = getattr(data, 'y', None)
        else:
            true = getattr(data[self.target_ntype], 'y', None)
        return logits, true

__all__ = ['HGTNet']
