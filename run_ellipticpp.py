import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, add_self_loops

# 경로
DATA_ROOT = "/local_datasets/ellipticpp"
RESULT_DIR = "/data/jhmun0206/results/fraudgt/ellipticpp"
os.makedirs(RESULT_DIR, exist_ok=True)

def find_col(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

def load_dataset(root):
    feats = pd.read_csv(os.path.join(root, "txs_features.csv"))
    labels = pd.read_csv(os.path.join(root, "txs_classes.csv"))
    edges = pd.read_csv(os.path.join(root, "txs_edgelist.csv"))

    id_feat  = find_col(feats.columns,  ["txId","tx_id","id"])
    id_label = find_col(labels.columns, ["txId","tx_id","id"])
    label_col= find_col(labels.columns, ["class","label","y"])

    df = pd.merge(feats, labels[[id_label, label_col]],
                  left_on=id_feat, right_on=id_label, how="inner")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in [id_feat, id_label, label_col]]
    df[feat_cols] = df[feat_cols].fillna(0)

    # Tensor 변환 + 표준화(z-score)
    x = torch.tensor(df[feat_cols].to_numpy(), dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std  = x.std(dim=0,  keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    y_raw = df[label_col].astype(str).tolist()
    uniq = sorted(set(y_raw))
    mapping = {c: i for i, c in enumerate(uniq)}
    y = torch.tensor([mapping[c] for c in y_raw], dtype=torch.long)

    # id -> index 매핑
    id_list = df[id_feat].astype(int).tolist()
    id2idx = {int(v): i for i, v in enumerate(id_list)}

    src_col = find_col(edges.columns, ["src","source","txId1","tx1"])
    dst_col = find_col(edges.columns, ["dst","target","txId2","tx2"])
    src, dst = [], []
    for s, d in zip(edges[src_col], edges[dst_col]):
        s = int(s); d = int(d)
        if s in id2idx and d in id2idx:
            src.append(id2idx[s]); dst.append(id2idx[d])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 무향 + self-loop
    edge_index = to_undirected(edge_index, num_nodes=len(df))
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(df))

    print(f"[INFO] Loaded {len(df)} nodes, {edge_index.size(1)} edges, {len(uniq)} classes")
    return Data(x=x, y=y, edge_index=edge_index), len(uniq)

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def main():
    data, num_classes = load_dataset(DATA_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GCN(data.num_features, 256, num_classes, 0.5).to(device)

    # 클래스 가중치 (불균형 보정)
    counts = torch.bincount(data.y)
    weights = (counts.sum() / (counts.float().clamp_min(1) * len(counts))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    N = data.num_nodes
    perm = torch.randperm(N, device=device)
    n_train, n_val = int(0.8 * N), int(0.1 * N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    print(f"[INFO] device={device}, classes={num_classes}")

    EPOCHS = 100
    for epoch in range(1, EPOCHS + 1):
        model.train()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
            acc_train = (pred[train_idx] == data.y[train_idx]).float().mean().item()
            acc_val   = (pred[val_idx]   == data.y[val_idx]).float().mean().item()
            acc_test  = (pred[test_idx]  == data.y[test_idx]).float().mean().item()

        if epoch % 5 == 0 or epoch <= 5:
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} train={acc_train:.4f} val={acc_val:.4f} test={acc_test:.4f}")

    save_path = os.path.join(RESULT_DIR, "ellipticpp_gnn.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model saved to {save_path}")

if __name__ == "__main__":
    main()
