import os
from typing import Any
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, HeteroData
from fraudGT.graphgym.config import cfg


def find_col(cols, names):
    """컬럼 이름 찾기 헬퍼 함수"""
    for n in names:
        if n in cols:
            return n
    return None


class EllipticPPPyG_TxWallet(InMemoryDataset):
    """
    Elliptic++ 데이터셋의 실제 wallet 정보를 활용한 tx+wallet 이종 그래프
    
    [Node Types]
    - 'tx': 트랜잭션 노드 (레이블 있음)
    - 'wallet': 지갑/주소 노드 (레이블 없음, 구조 정보만 제공)
    
    [Edge Types]
    - ('tx', 'fwd', 'tx'): txs_edgelist.csv (순방향)
    - ('tx', 'rev', 'tx'): 위 엣지의 reverse
    - ('tx', 'to', 'wallet'): TxAddr_edgelist.csv
    - ('wallet', 'from', 'tx'): AddrTx_edgelist.csv
    - ('wallet', 'link', 'wallet'): AddrAddr_edgelist.csv (옵션)
    """
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        # InMemoryDataset의 __init__은 process()를 호출하지만,
        # 저장된 파일이 있으면 process()를 건너뜀
        # 따라서 __init__에서 직접 처리하여 확실하게 self.data를 설정
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # 저장된 파일이 있으면 로드, 없으면 process() 호출 후 저장
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
            print("[INFO] 저장된 데이터셋 파일 로드 중...")
            self.data, self.slices = torch.load(processed_path)
            print("[INFO] 데이터셋 로드 완료")
        else:
            # process() 호출하여 데이터 생성 및 저장
            print("[INFO] 데이터셋 파일이 없어 process() 실행...")
            self.process()
            # process() 완료 후 저장된 파일 로드
            if os.path.exists(processed_path):
                self.data, self.slices = torch.load(processed_path)
                print("[INFO] 저장된 데이터셋 파일 로드 완료")
            else:
                raise RuntimeError(f"데이터셋 저장 실패: {processed_path}")
    
    @property
    def raw_file_names(self):
        """필요한 원본 CSV 파일 목록"""
        return [
            'txs_features.csv',
            'txs_classes.csv',
            'txs_edgelist.csv',
            'wallets_features.csv',
            'wallets_classes.csv',  # 옵션 (사용 안 할 수도 있음)
            'TxAddr_edgelist.csv',
            'AddrTx_edgelist.csv',
            'AddrAddr_edgelist.csv',  # 옵션
        ]
    
    @property
    def processed_file_names(self):
        return ['ellipticpp_tx_wallet.pt']
    
    def process(self):
        """이종 그래프 생성"""
        # PyG InMemoryDataset가 자동으로 설정해주는 raw 디렉토리
        raw_dir = self.raw_dir  # == os.path.join(self.root, 'raw')
        
        print("[INFO] Elliptic++ Tx+Wallet 이종 그래프 생성 시작...")
        
        # ========== 1. CSV 파일 읽기 ==========
        print("[INFO] CSV 파일 로딩 중...")
        
        # Transaction 파일
        txs_feats = pd.read_csv(os.path.join(raw_dir, 'txs_features.csv'))
        txs_labels = pd.read_csv(os.path.join(raw_dir, 'txs_classes.csv'))
        txs_edges = pd.read_csv(os.path.join(raw_dir, 'txs_edgelist.csv'))
        
        # Wallet 파일 (옵션: 없을 수도 있음)
        wallets_feats_path = os.path.join(raw_dir, 'wallets_features.csv')
        wallets_feats = None
        if os.path.exists(wallets_feats_path):
            wallets_feats = pd.read_csv(wallets_feats_path)
            print(f"  - wallets_features.csv 로드 완료: {len(wallets_feats)}개")
        else:
            print(f"  - wallets_features.csv 없음 (건너뜀)")
        
        # Edge 파일들
        txaddr_edges_path = os.path.join(raw_dir, 'TxAddr_edgelist.csv')
        addrtx_edges_path = os.path.join(raw_dir, 'AddrTx_edgelist.csv')
        addraddr_edges_path = os.path.join(raw_dir, 'AddrAddr_edgelist.csv')
        
        txaddr_edges = None
        if os.path.exists(txaddr_edges_path):
            txaddr_edges = pd.read_csv(txaddr_edges_path)
            print(f"  - TxAddr_edgelist.csv 로드 완료: {len(txaddr_edges)}개")
            print(f"    [DEBUG] TxAddr columns: {list(txaddr_edges.columns)}")
            if len(txaddr_edges) > 0:
                print(f"    [DEBUG] TxAddr 첫 3행:\n{txaddr_edges.head(3)}")
        
        addrtx_edges = None
        if os.path.exists(addrtx_edges_path):
            addrtx_edges = pd.read_csv(addrtx_edges_path)
            print(f"  - AddrTx_edgelist.csv 로드 완료: {len(addrtx_edges)}개")
            print(f"    [DEBUG] AddrTx columns: {list(addrtx_edges.columns)}")
            if len(addrtx_edges) > 0:
                print(f"    [DEBUG] AddrTx 첫 3행:\n{addrtx_edges.head(3)}")
        
        addraddr_edges = None
        if os.path.exists(addraddr_edges_path):
            addraddr_edges = pd.read_csv(addraddr_edges_path)
            print(f"  - AddrAddr_edgelist.csv 로드 완료: {len(addraddr_edges)}개")
            print(f"    [DEBUG] AddrAddr columns: {list(addraddr_edges.columns)}")
            if len(addraddr_edges) > 0:
                print(f"    [DEBUG] AddrAddr 첫 3행:\n{addraddr_edges.head(3)}")
        
        # ========== 2. 컬럼 이름 찾기 ==========
        tx_id_feat = find_col(txs_feats.columns, ['txId', 'tx_id', 'id', 'Id'])
        tx_id_label = find_col(txs_labels.columns, ['txId', 'tx_id', 'id', 'Id'])
        tx_label_col = find_col(txs_labels.columns, ['class', 'label', 'y'])
        
        tx_src_col = find_col(txs_edges.columns, ['src', 'source', 'txId1', 'tx1', 'from'])
        tx_dst_col = find_col(txs_edges.columns, ['dst', 'target', 'txId2', 'tx2', 'to'])
        
        # ========== 3. Transaction ID 매핑 (0..N-1) ==========
        print("[INFO] Transaction ID 매핑 중...")
        df_tx = pd.merge(
            txs_feats, 
            txs_labels[[tx_id_label, tx_label_col]],
            left_on=tx_id_feat,
            right_on=tx_id_label,
            how='inner'
        )
        df_tx = df_tx.dropna(subset=[tx_label_col]).reset_index(drop=True)
        
        tx_id_list = df_tx[tx_id_feat].astype(int).tolist()
        tx_id2idx = {int(v): i for i, v in enumerate(tx_id_list)}
        num_txs = len(tx_id2idx)
        print(f"  - Transaction 노드: {num_txs:,}개")
        
        # ========== 4. Wallet ID 매핑 (0..M-1) ==========
        print("[INFO] Wallet ID 매핑 중...")
        wallet_ids = set()
        
        # Wallet ID는 문자열로 유지 (비트코인 주소 등)
        # 1) wallets_features.csv에서 wallet ID 수집
        if wallets_feats is not None:
            wallet_id_col = find_col(wallets_feats.columns, ['walletId', 'wallet', 'addr', 'address', 'addrId', 'id', 'Id'])
            if wallet_id_col:
                wallets_feats[wallet_id_col] = wallets_feats[wallet_id_col].astype(str)
                wallet_ids.update(wallets_feats[wallet_id_col].tolist())
        
        # 2) TxAddr_edgelist에서 wallet ID 수집
        if txaddr_edges is not None:
            wallet_col = find_col(txaddr_edges.columns, ['output_address', 'wallet', 'walletId', 'addr', 'address', 'addrId', 'to'])
            if wallet_col:
                txaddr_edges[wallet_col] = txaddr_edges[wallet_col].astype(str)
                wallet_ids.update(txaddr_edges[wallet_col].tolist())
        
        # 3) AddrTx_edgelist에서 wallet ID 수집
        if addrtx_edges is not None:
            wallet_col = find_col(addrtx_edges.columns, ['input_address', 'wallet', 'walletId', 'addr', 'address', 'addrId', 'from'])
            if wallet_col:
                addrtx_edges[wallet_col] = addrtx_edges[wallet_col].astype(str)
                wallet_ids.update(addrtx_edges[wallet_col].tolist())
        
        # 4) AddrAddr_edgelist에서 wallet ID 수집 (src, dst 둘 다)
        if addraddr_edges is not None:
            wallet_src_col = find_col(addraddr_edges.columns, ['input_address', 'src', 'source', 'wallet1', 'addr1', 'from'])
            wallet_dst_col = find_col(addraddr_edges.columns, ['output_address', 'dst', 'target', 'wallet2', 'addr2', 'to'])
            if wallet_src_col:
                addraddr_edges[wallet_src_col] = addraddr_edges[wallet_src_col].astype(str)
                wallet_ids.update(addraddr_edges[wallet_src_col].tolist())
            if wallet_dst_col:
                addraddr_edges[wallet_dst_col] = addraddr_edges[wallet_dst_col].astype(str)
                wallet_ids.update(addraddr_edges[wallet_dst_col].tolist())
        
        # set → list → dict로 변환 (문자열 ID → 인덱스 매핑)
        wallet_id_list = sorted(list(wallet_ids))
        wallet_id2idx = {wid: i for i, wid in enumerate(wallet_id_list)}
        num_wallets = len(wallet_id2idx)
        print(f"  - Wallet 노드: {num_wallets:,}개")
        print(f"    [DEBUG] Example wallet IDs (mapping keys): {list(wallet_id2idx.keys())[:5]}")
        print(f"    [DEBUG] Wallet ID type: {type(list(wallet_id2idx.keys())[0]) if len(wallet_id2idx) > 0 else 'N/A'}")
        
        # ========== 5. Transaction Features & Labels ==========
        print("[INFO] Transaction 특징 및 레이블 처리 중...")
        feat_cols = [c for c in df_tx.columns if c not in [tx_id_feat, tx_id_label, tx_label_col]]
        df_tx[feat_cols] = df_tx[feat_cols].fillna(0)
        
        # Tensor 변환 + 표준화
        tx_x = torch.tensor(df_tx[feat_cols].to_numpy(), dtype=torch.float32)
        mean = tx_x.mean(dim=0, keepdim=True)
        std = tx_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        tx_x = (tx_x - mean) / std
        
        # Label 처리
        y_raw = df_tx[tx_label_col].astype(str).tolist()
        uniq = sorted(set(y_raw))
        label_mapping = {c: i for i, c in enumerate(uniq)}
        tx_y = torch.tensor([label_mapping[c] for c in y_raw], dtype=torch.long)
        print(f"  - 특징 차원: {tx_x.shape[1]}")
        print(f"  - 클래스 수: {len(uniq)}")
        
        # ========== 6. Train/Val/Test Mask 구성 ==========
        print("[INFO] Train/Val/Test split 생성 중...")
        # timestep 기반 split (Elliptic++의 표준 방식)
        # timestep 컬럼이 있으면 사용, 없으면 랜덤 split
        timestep_col = find_col(df_tx.columns, ['timestep', 'time', 'time_step'])
        if timestep_col:
            timesteps = df_tx[timestep_col].values
            # 일반적으로 초기 70%는 train, 다음 15%는 val, 나머지는 test
            unique_timesteps = sorted(set(timesteps))
            n_train = int(len(unique_timesteps) * 0.7)
            n_val = int(len(unique_timesteps) * 0.15)
            train_timesteps = set(unique_timesteps[:n_train])
            val_timesteps = set(unique_timesteps[n_train:n_train+n_val])
            test_timesteps = set(unique_timesteps[n_train+n_val:])
            
            train_mask = torch.tensor([t in train_timesteps for t in timesteps], dtype=torch.bool)
            val_mask = torch.tensor([t in val_timesteps for t in timesteps], dtype=torch.bool)
            test_mask = torch.tensor([t in test_timesteps for t in timesteps], dtype=torch.bool)
            print(f"  - Train: {train_mask.sum().item():,}개")
            print(f"  - Val: {val_mask.sum().item():,}개")
            print(f"  - Test: {test_mask.sum().item():,}개")
        else:
            # 랜덤 split
            from sklearn.model_selection import train_test_split
            indices = np.arange(num_txs)
            train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
            
            train_mask = torch.zeros(num_txs, dtype=torch.bool)
            val_mask = torch.zeros(num_txs, dtype=torch.bool)
            test_mask = torch.zeros(num_txs, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            print(f"  - Train: {train_mask.sum().item():,}개 (랜덤 split)")
            print(f"  - Val: {val_mask.sum().item():,}개")
            print(f"  - Test: {test_mask.sum().item():,}개")
        
        # ========== 7. Wallet Features ==========
        print("[INFO] Wallet 특징 처리 중...")
        if wallets_feats is not None:
            wallet_id_col = find_col(wallets_feats.columns, ['walletId', 'wallet', 'addr', 'address', 'addrId', 'id', 'Id'])
            if wallet_id_col:
                # wallet_id_col을 문자열로 변환
                wallets_feats[wallet_id_col] = wallets_feats[wallet_id_col].astype(str)
                
                # 매핑된 wallet만 사용
                wallets_feats = wallets_feats[wallets_feats[wallet_id_col].isin(wallet_id_list)].copy()
                
                # wallet_id2idx 기준으로 인덱스 정렬
                wallets_feats['wallet_idx'] = wallets_feats[wallet_id_col].map(wallet_id2idx)
                wallets_feats = wallets_feats.sort_values('wallet_idx')
                
                feat_cols_w = [c for c in wallets_feats.columns if c not in [wallet_id_col, 'wallet_idx']]
                wallets_feats[feat_cols_w] = wallets_feats[feat_cols_w].fillna(0)
                
                # 벡터화된 성능 개선: pandas DataFrame을 직접 사용하여 벡터 연산
                # 중복 제거: wallet_id_col 기준으로 첫 번째 행만 유지
                print("    Wallet feature 벡터화 처리 중...")
                wallets_feats_dedup = wallets_feats.drop_duplicates(subset=[wallet_id_col], keep='first')
                
                # wallet_id_col을 인덱스로 설정하여 빠른 조회 가능하게 함
                wallets_feats_dedup = wallets_feats_dedup.set_index(wallet_id_col)
                
                # wallet_id_list 순서대로 reindex (없는 것은 NaN으로 채워짐)
                total_wallets = len(wallet_id_list)
                wallet_features_df = wallets_feats_dedup.reindex(
                    wallet_id_list, 
                    fill_value=0.0  # 없는 wallet은 0으로 채움
                )
                
                # feat_cols_w만 선택하여 numpy 배열로 변환 (벡터화된 연산)
                wallet_features_array = wallet_features_df[feat_cols_w].values.astype(np.float32)
                
                # torch tensor로 변환 (한 번만 복사)
                wallet_features_tensor = torch.from_numpy(wallet_features_array).float()
                
                print(f"    벡터화 처리 완료: {total_wallets:,}개 wallet × {len(feat_cols_w)}차원")
                
                # 표준화
                print("    Wallet feature 표준화 중...")
                mean_w = wallet_features_tensor.mean(dim=0, keepdim=True)
                std_w = wallet_features_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
                wallet_features_tensor = (wallet_features_tensor - mean_w) / std_w
                print("    표준화 완료")
                
                # 통계 출력 (벡터화된 계산)
                # wallets_feats_dedup에 있는 wallet = feature가 있는 wallet
                num_wallets_with_feat = len(wallets_feats_dedup.index.intersection(wallet_id_list))
                num_wallets_zero = total_wallets - num_wallets_with_feat
                print(f"  - Wallet 특징 차원: {wallet_features_tensor.shape[1]}")
                print(f"  - Feature 있는 wallet: {num_wallets_with_feat:,}개")
                if num_wallets_zero > 0:
                    print(f"  - Feature 없는 wallet (zero로 채움): {num_wallets_zero:,}개")
                
                wallet_x = wallet_features_tensor
            else:
                # wallet_id_col을 찾을 수 없으면 랜덤 feature 생성
                wallet_x = torch.randn(num_wallets, tx_x.shape[1], dtype=torch.float32)
                print(f"  - Wallet 특징: 랜덤 생성 ({wallet_x.shape[1]}차원)")
        else:
            # wallets_features.csv가 없으면 tx와 동일한 차원의 랜덤 feature 생성
            wallet_x = torch.randn(num_wallets, tx_x.shape[1], dtype=torch.float32)
            print(f"  - Wallet 특징: 랜덤 생성 ({wallet_x.shape[1]}차원)")
        
        # 인덱스 일관성 확인
        assert wallet_x.shape[0] == num_wallets, f"Wallet feature tensor size mismatch: {wallet_x.shape[0]} != {num_wallets}"
        
        # ========== 8. Edge Index 생성 ==========
        print("[INFO] Edge Index 생성 중...")
        hetero = HeteroData()
        
        # Transaction 노드
        hetero['tx'].x = tx_x
        hetero['tx'].y = tx_y
        hetero['tx'].train_mask = train_mask
        hetero['tx'].val_mask = val_mask
        hetero['tx'].test_mask = test_mask
        
        # Wallet 노드
        hetero['wallet'].x = wallet_x
        
        # ('tx', 'fwd', 'tx') 엣지
        print("    ('tx', 'fwd', 'tx') 엣지 처리 중...")
        tx_src_list, tx_dst_list = [], []
        total_tx_edges = len(txs_edges)
        log_interval = max(10000, total_tx_edges // 20)  # 최소 1만개당, 최대 20번 출력
        
        for idx, (s, d) in enumerate(zip(txs_edges[tx_src_col], txs_edges[tx_dst_col])):
            if (idx + 1) % log_interval == 0:
                print(f"      처리 중: {idx + 1:,}/{total_tx_edges:,} ({100 * (idx + 1) / total_tx_edges:.1f}%)")
            
            s, d = int(s), int(d)
            if s in tx_id2idx and d in tx_id2idx:
                tx_src_list.append(tx_id2idx[s])
                tx_dst_list.append(tx_id2idx[d])
        
        if len(tx_src_list) > 0:
            hetero[('tx', 'fwd', 'tx')].edge_index = torch.tensor(
                [tx_src_list, tx_dst_list], dtype=torch.long
            ).contiguous()
            print(f"  - ('tx', 'fwd', 'tx'): {len(tx_src_list):,}개")
        
        # ('tx', 'rev', 'tx') 엣지 (reverse)
        if ('tx', 'fwd', 'tx') in hetero.edge_types:
            hetero[('tx', 'rev', 'tx')].edge_index = hetero[('tx', 'fwd', 'tx')].edge_index.flip(0).contiguous()
            print(f"  - ('tx', 'rev', 'tx'): {hetero[('tx', 'rev', 'tx')].edge_index.size(1):,}개")
        
        # ('tx', 'to', 'wallet') 엣지
        if txaddr_edges is not None:
            print("    ('tx', 'to', 'wallet') 엣지 처리 중...")
            tx_col = find_col(txaddr_edges.columns, ['tx', 'txId', 'tx_id', 'from'])
            wallet_col = find_col(txaddr_edges.columns, ['output_address', 'wallet', 'walletId', 'addr', 'address', 'addrId', 'to'])
            print(f"    [DEBUG] tx_col (TxAddr): {tx_col}")
            print(f"    [DEBUG] wallet_col (TxAddr): {wallet_col}")
            if tx_col and wallet_col:
                if len(txaddr_edges) > 0:
                    print(f"    [DEBUG] Sample TxAddr wallet IDs: {txaddr_edges[wallet_col].head(5).tolist()}")
                    print(f"    [DEBUG] Sample TxAddr wallet ID types: {[type(x) for x in txaddr_edges[wallet_col].head(3).tolist()]}")
                # tx id는 int로 변환, wallet id는 문자열로 유지
                txaddr_edges[tx_col] = txaddr_edges[tx_col].astype(int)
                # wallet_col은 이미 astype(str)로 변환됨
                
                # 매핑 적용
                print(f"      매핑 적용 중... (총 {len(txaddr_edges):,}개 엣지)")
                txaddr_edges['tx_idx'] = txaddr_edges[tx_col].map(tx_id2idx)
                txaddr_edges['wallet_idx'] = txaddr_edges[wallet_col].map(wallet_id2idx)
                
                # 유효한 매핑만 필터링
                valid_mask = txaddr_edges['tx_idx'].notna() & txaddr_edges['wallet_idx'].notna()
                txaddr_valid = txaddr_edges[valid_mask]
                print(f"      [DEBUG] Tx→Wallet valid edges: {valid_mask.sum():,}개")
                print(f"      [DEBUG] tx_idx notna: {txaddr_edges['tx_idx'].notna().sum():,}개")
                print(f"      [DEBUG] wallet_idx notna: {txaddr_edges['wallet_idx'].notna().sum():,}개")
                if valid_mask.sum() == 0:
                    print(f"      [WARNING] 매핑 실패! 샘플 확인:")
                    print(f"        - tx_col 값 샘플: {txaddr_edges[tx_col].head(5).tolist()}")
                    print(f"        - wallet_col 값 샘플: {txaddr_edges[wallet_col].head(5).tolist()}")
                    print(f"        - tx_id2idx에 있는지 확인: {txaddr_edges[tx_col].head(3).isin([int(k) for k in tx_id2idx.keys()]).tolist()}")
                    print(f"        - wallet_id2idx에 있는지 확인: {txaddr_edges[wallet_col].head(3).isin(list(wallet_id2idx.keys())).tolist()}")
                print(f"      유효한 엣지: {len(txaddr_valid):,}개")
                
                if len(txaddr_valid) > 0:
                    src_tx = txaddr_valid['tx_idx'].astype(int).values
                    dst_wallet = txaddr_valid['wallet_idx'].astype(int).values
                    
                    edge_index_tx_to_wallet = torch.tensor(
                        [src_tx, dst_wallet], dtype=torch.long
                    ).contiguous()
                    
                    hetero[('tx', 'to', 'wallet')].edge_index = edge_index_tx_to_wallet
                    print(f"  - ('tx', 'to', 'wallet'): {len(src_tx):,}개")
        
        # ('wallet', 'from', 'tx') 엣지
        if addrtx_edges is not None:
            print("    ('wallet', 'from', 'tx') 엣지 처리 중...")
            wallet_col = find_col(addrtx_edges.columns, ['input_address', 'wallet', 'walletId', 'addr', 'address', 'addrId', 'from'])
            tx_col = find_col(addrtx_edges.columns, ['tx', 'txId', 'tx_id', 'to'])
            print(f"    [DEBUG] wallet_col (AddrTx): {wallet_col}")
            print(f"    [DEBUG] tx_col (AddrTx): {tx_col}")
            if wallet_col and tx_col:
                if len(addrtx_edges) > 0:
                    print(f"    [DEBUG] Sample AddrTx wallet IDs: {addrtx_edges[wallet_col].head(5).tolist()}")
                # tx id는 int로 변환, wallet id는 문자열로 유지
                addrtx_edges[tx_col] = addrtx_edges[tx_col].astype(int)
                # wallet_col은 이미 astype(str)로 변환됨
                
                # 매핑 적용
                print(f"      매핑 적용 중... (총 {len(addrtx_edges):,}개 엣지)")
                addrtx_edges['wallet_idx'] = addrtx_edges[wallet_col].map(wallet_id2idx)
                addrtx_edges['tx_idx'] = addrtx_edges[tx_col].map(tx_id2idx)
                
                # 유효한 매핑만 필터링
                valid_mask = addrtx_edges['wallet_idx'].notna() & addrtx_edges['tx_idx'].notna()
                addrtx_valid = addrtx_edges[valid_mask]
                print(f"      [DEBUG] Wallet→Tx valid edges: {valid_mask.sum():,}개")
                print(f"      [DEBUG] wallet_idx notna: {addrtx_edges['wallet_idx'].notna().sum():,}개")
                print(f"      [DEBUG] tx_idx notna: {addrtx_edges['tx_idx'].notna().sum():,}개")
                if valid_mask.sum() == 0:
                    print(f"      [WARNING] 매핑 실패! 샘플 확인:")
                    print(f"        - wallet_col 값 샘플: {addrtx_edges[wallet_col].head(5).tolist()}")
                    print(f"        - tx_col 값 샘플: {addrtx_edges[tx_col].head(5).tolist()}")
                print(f"      유효한 엣지: {len(addrtx_valid):,}개")
                
                if len(addrtx_valid) > 0:
                    src_wallet = addrtx_valid['wallet_idx'].astype(int).values
                    dst_tx = addrtx_valid['tx_idx'].astype(int).values
                    
                    edge_index_wallet_to_tx = torch.tensor(
                        [src_wallet, dst_tx], dtype=torch.long
                    ).contiguous()
                    
                    hetero[('wallet', 'from', 'tx')].edge_index = edge_index_wallet_to_tx
                    print(f"  - ('wallet', 'from', 'tx'): {len(src_wallet):,}개")
        
        # ('wallet', 'link', 'wallet') 엣지 (옵션)
        if addraddr_edges is not None:
            print("    ('wallet', 'link', 'wallet') 엣지 처리 중...")
            wallet_src_col = find_col(addraddr_edges.columns, ['input_address', 'src', 'source', 'wallet1', 'addr1', 'from'])
            wallet_dst_col = find_col(addraddr_edges.columns, ['output_address', 'dst', 'target', 'wallet2', 'addr2', 'to'])
            print(f"    [DEBUG] wallet_src_col (AddrAddr): {wallet_src_col}")
            print(f"    [DEBUG] wallet_dst_col (AddrAddr): {wallet_dst_col}")
            if wallet_src_col and wallet_dst_col:
                if len(addraddr_edges) > 0:
                    print(f"    [DEBUG] Sample AddrAddr src wallet IDs: {addraddr_edges[wallet_src_col].head(5).tolist()}")
                    print(f"    [DEBUG] Sample AddrAddr dst wallet IDs: {addraddr_edges[wallet_dst_col].head(5).tolist()}")
                # wallet id는 문자열로 유지 (이미 astype(str)로 변환됨)
                # 샘플링 옵션 확인
                sample_ratio = getattr(cfg.dataset, 'wallet_edge_sample_ratio', 1.0) if hasattr(cfg, 'dataset') else 1.0
                max_wallet_edges = getattr(cfg.dataset, 'max_wallet_nodes', None) if hasattr(cfg, 'dataset') else None
                
                print(f"      매핑 적용 중... (총 {len(addraddr_edges):,}개 엣지)")
                # 매핑 적용
                addraddr_edges['src_idx'] = addraddr_edges[wallet_src_col].map(wallet_id2idx)
                addraddr_edges['dst_idx'] = addraddr_edges[wallet_dst_col].map(wallet_id2idx)
                
                # 유효한 매핑만 필터링
                valid_mask = addraddr_edges['src_idx'].notna() & addraddr_edges['dst_idx'].notna()
                addraddr_valid = addraddr_edges[valid_mask].copy()
                print(f"      [DEBUG] Wallet↔Wallet valid edges: {valid_mask.sum():,}개")
                print(f"      [DEBUG] src_idx notna: {addraddr_edges['src_idx'].notna().sum():,}개")
                print(f"      [DEBUG] dst_idx notna: {addraddr_edges['dst_idx'].notna().sum():,}개")
                if valid_mask.sum() == 0:
                    print(f"      [WARNING] 매핑 실패! 샘플 확인:")
                    print(f"        - wallet_src_col 값 샘플: {addraddr_edges[wallet_src_col].head(5).tolist()}")
                    print(f"        - wallet_dst_col 값 샘플: {addraddr_edges[wallet_dst_col].head(5).tolist()}")
                print(f"      유효한 엣지: {len(addraddr_valid):,}개")
                
                # 샘플링
                if sample_ratio < 1.0:
                    print(f"      샘플링 적용: {sample_ratio * 100:.1f}%")
                    addraddr_valid = addraddr_valid.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
                    print(f"      샘플링 후: {len(addraddr_valid):,}개")
                
                if len(addraddr_valid) > 0:
                    # 엣지 수 제한
                    if max_wallet_edges and len(addraddr_valid) > max_wallet_edges:
                        print(f"      엣지 수 제한: {max_wallet_edges:,}개로 제한")
                        addraddr_valid = addraddr_valid.head(max_wallet_edges)
                    
                    src_wallet = addraddr_valid['src_idx'].astype(int).values
                    dst_wallet = addraddr_valid['dst_idx'].astype(int).values
                    
                    edge_index_wallet_wallet = torch.tensor(
                        [src_wallet, dst_wallet], dtype=torch.long
                    ).contiguous()
                    
                    hetero[('wallet', 'link', 'wallet')].edge_index = edge_index_wallet_wallet
                    print(f"  - ('wallet', 'link', 'wallet'): {len(src_wallet):,}개")
        
        # ========== 9. 통계 출력 ==========
        print("\n" + "=" * 80)
        print("[INFO] 이종 그래프 생성 완료 - 통계 요약")
        print("=" * 80)
        print(f"[INFO] 노드 통계:")
        print(f"  - 트랜잭션 노드 (tx): {num_txs:,}개")
        print(f"  - 지갑 노드 (wallet): {num_wallets:,}개")
        print(f"  - 전체 노드: {num_txs + num_wallets:,}개")
        print(f"[INFO] 엣지 통계:")
        total_edges = 0
        for edge_type in hetero.edge_types:
            num_edges = hetero[edge_type].edge_index.size(1)
            total_edges += num_edges
            print(f"  - {edge_type}: {num_edges:,}개")
        print(f"  - 전체 엣지: {total_edges:,}개")
        print(f"[INFO] 엣지 타입: {list(hetero.edge_types)}")
        print("=" * 80)
        
        # InMemoryDataset 규약에 맞게 (data, slices) 저장
        print("[INFO] 데이터셋 저장 중...")
        data_list = [hetero]
        data, slices = self.collate(data_list)
        
        # 저장 경로 확인
        save_path = self.processed_paths[0]
        print(f"[INFO] 저장 경로: {save_path}")
        
        # 디렉토리 생성 (필요시)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 저장 실행
        torch.save((data, slices), save_path)
        print(f"[INFO] 데이터셋 저장 완료: {save_path}")
        
        # 저장 확인
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            print(f"[INFO] 저장된 파일 크기: {file_size:.2f} MB")
        else:
            print(f"[ERROR] 저장 실패: 파일이 생성되지 않았습니다!")
        
        # return은 필요 없음 (InMemoryDataset이 무시함)

