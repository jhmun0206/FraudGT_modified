#!/usr/bin/env python
"""
FN/FP 샘플 분석 스크립트

사용법:
    python analyze_fn_fp.py /data/jhmun0206/results/fraudgt/ellipticpp_hetero_cluster/hetero-cluster/config/0
    python analyze_fn_fp.py <결과_디렉토리>
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fraudGT.graphgym.config import cfg, set_cfg
    from fraudGT.loader.master_loader import load_dataset_master
except ImportError as e:
    print(f"[WARNING] fraudGT 모듈을 import할 수 없습니다: {e}")
    print("데이터셋 로딩 없이 분석을 진행합니다.")

def load_predictions_and_labels(result_dir):
    """예측 결과와 레이블 로드"""
    result_dir = Path(result_dir)
    
    pred_file = result_dir / 'test_predictions.npy'
    label_file = result_dir / 'test_labels.npy'
    
    if not pred_file.exists():
        raise FileNotFoundError(f"test_predictions.npy를 찾을 수 없습니다: {pred_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"test_labels.npy를 찾을 수 없습니다: {label_file}")
    
    predictions = np.load(pred_file)
    labels = np.load(label_file)
    
    # 예측 클래스 추출
    if len(predictions.shape) == 2:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    return predictions, labels, pred_classes

def extract_fn_fp_indices(labels, pred_classes):
    """FN과 FP 인덱스 추출"""
    # FN: 실제 fraud(2)인데 다른 클래스로 예측
    fn_mask = (labels == 2) & (pred_classes != 2)
    fn_indices = np.where(fn_mask)[0]
    
    # FP: 실제 normal(0) 또는 unknown(1)인데 fraud(2)로 예측
    fp_mask = (labels != 2) & (pred_classes == 2)
    fp_indices = np.where(fp_mask)[0]
    
    return fn_indices, fp_indices

def analyze_fn_fp(result_dir, dataset_dir=None):
    """FN/FP 분석"""
    result_dir = Path(result_dir)
    
    print("=" * 80)
    print("FN/FP 샘플 분석")
    print("=" * 80)
    print(f"결과 디렉토리: {result_dir}")
    print()
    
    # 1. 예측 결과 로드
    print("[1] 예측 결과 로드 중...")
    predictions, labels, pred_classes = load_predictions_and_labels(result_dir)
    
    print(f"  예측 shape: {predictions.shape}")
    print(f"  레이블 shape: {labels.shape}")
    print(f"  클래스 분포 (실제): {np.bincount(labels.astype(int))}")
    print(f"  클래스 분포 (예측): {np.bincount(pred_classes.astype(int))}")
    
    # 2. FN/FP 추출
    print("\n[2] FN/FP 추출 중...")
    fn_indices, fp_indices = extract_fn_fp_indices(labels, pred_classes)
    
    print(f"  FN (False Negative): {len(fn_indices)}개")
    print(f"    - 실제 fraud(2)인데 다른 클래스로 예측")
    print(f"    - 예측 분포: {np.bincount(pred_classes[fn_indices].astype(int))}")
    
    print(f"  FP (False Positive): {len(fp_indices)}개")
    print(f"    - 실제 normal(0) 또는 unknown(1)인데 fraud(2)로 예측")
    print(f"    - 실제 분포: {np.bincount(labels[fp_indices].astype(int))}")
    
    # 3. 원본 데이터셋과 연결 (가능한 경우)
    fn_details = []
    fp_details = []
    
    if dataset_dir and 'fraudGT' in sys.modules:
        try:
            print("\n[3] 원본 데이터셋 로드 중...")
            # Config 로드 시도
            config_file = result_dir.parent.parent.parent / 'config.yaml'
            if config_file.exists():
                set_cfg(cfg)
                cfg.merge_from_file(str(config_file))
                cfg.freeze()
            
            dataset = load_dataset_master(
                format=cfg.dataset.format,
                name=cfg.dataset.name,
                dataset_dir=dataset_dir
            )
            
            data = dataset[0] if hasattr(dataset, '__getitem__') else dataset.data
            
            # 이종 그래프인 경우 tx 노드만 사용
            if hasattr(data, 'node_types') and 'tx' in data.node_types:
                tx_data = data['tx']
                tx_features = tx_data.x
                tx_labels = tx_data.y
                if hasattr(tx_data, 'test_mask'):
                    test_mask = tx_data.test_mask
                    test_indices = torch.where(test_mask)[0].cpu().numpy()
                else:
                    test_indices = np.arange(len(tx_labels))
            else:
                tx_features = data.x
                tx_labels = data.y
                if hasattr(data, 'test_mask'):
                    test_mask = data.test_mask
                    test_indices = torch.where(test_mask)[0].cpu().numpy()
                else:
                    test_indices = np.arange(len(tx_labels))
            
            print(f"  데이터셋 로드 완료: {len(tx_features)}개 트랜잭션")
            
            # FN 상세 정보
            if len(fn_indices) > 0:
                print(f"\n[4] FN 샘플 상세 정보 추출 중...")
                for idx in fn_indices[:min(100, len(fn_indices))]:  # 최대 100개만
                    if idx < len(test_indices):
                        tx_idx = test_indices[idx]
                        if tx_idx < len(tx_features):
                            fn_details.append({
                                'test_index': idx,
                                'node_index': int(tx_idx),
                                'true_label': int(labels[idx]),
                                'predicted_label': int(pred_classes[idx]),
                                'prediction_confidence': float(predictions[idx, pred_classes[idx]]) if len(predictions.shape) == 2 else 0.0
                            })
            
            # FP 상세 정보
            if len(fp_indices) > 0:
                print(f"[5] FP 샘플 상세 정보 추출 중...")
                for idx in fp_indices[:min(100, len(fp_indices))]:  # 최대 100개만
                    if idx < len(test_indices):
                        tx_idx = test_indices[idx]
                        if tx_idx < len(tx_features):
                            fp_details.append({
                                'test_index': idx,
                                'node_index': int(tx_idx),
                                'true_label': int(labels[idx]),
                                'predicted_label': int(pred_classes[idx]),
                                'prediction_confidence': float(predictions[idx, pred_classes[idx]]) if len(predictions.shape) == 2 else 0.0
                            })
            
        except Exception as e:
            print(f"  [WARNING] 데이터셋 로드 실패: {e}")
            print("  인덱스 정보만 저장합니다.")
            # 인덱스만 저장
            for idx in fn_indices[:min(100, len(fn_indices))]:
                fn_details.append({
                    'test_index': int(idx),
                    'true_label': int(labels[idx]),
                    'predicted_label': int(pred_classes[idx]),
                    'prediction_confidence': float(predictions[idx, pred_classes[idx]]) if len(predictions.shape) == 2 else 0.0
                })
            for idx in fp_indices[:min(100, len(fp_indices))]:
                fp_details.append({
                    'test_index': int(idx),
                    'true_label': int(labels[idx]),
                    'predicted_label': int(pred_classes[idx]),
                    'prediction_confidence': float(predictions[idx, pred_classes[idx]]) if len(predictions.shape) == 2 else 0.0
                })
    else:
        print("\n[3] 데이터셋 정보 없이 인덱스만 저장합니다.")
        for idx in fn_indices[:min(100, len(fn_indices))]:
            fn_details.append({
                'test_index': int(idx),
                'true_label': int(labels[idx]),
                'predicted_label': int(pred_classes[idx]),
                'prediction_confidence': float(predictions[idx, pred_classes[idx]]) if len(predictions.shape) == 2 else 0.0
            })
        for idx in fp_indices[:min(100, len(fp_indices))]:
            fp_details.append({
                'test_index': int(idx),
                'true_label': int(labels[idx]),
                'predicted_label': int(pred_classes[idx]),
                'prediction_confidence': float(predictions[idx, pred_classes[idx]]) if len(predictions.shape) == 2 else 0.0
            })
    
    # 4. CSV 저장
    print("\n[6] 결과 저장 중...")
    fn_df = pd.DataFrame(fn_details)
    fp_df = pd.DataFrame(fp_details)
    
    fn_csv = result_dir / 'fn_samples.csv'
    fp_csv = result_dir / 'fp_samples.csv'
    
    fn_df.to_csv(fn_csv, index=False)
    fp_df.to_csv(fp_csv, index=False)
    
    print(f"  ✅ FN 샘플: {fn_csv} ({len(fn_df)}개)")
    print(f"  ✅ FP 샘플: {fp_csv} ({len(fp_df)}개)")
    
    # 5. 요약 출력
    print("\n" + "=" * 80)
    print("분석 요약")
    print("=" * 80)
    print(f"FN (False Negative): {len(fn_indices)}개")
    print(f"  - 실제 fraud를 놓친 경우")
    print(f"  - 상세 정보: {len(fn_df)}개 저장됨")
    
    print(f"\nFP (False Positive): {len(fp_indices)}개")
    print(f"  - 정상 거래를 잘못 의심한 경우")
    print(f"  - 상세 정보: {len(fp_df)}개 저장됨")
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)
    print(f"결과 파일:")
    print(f"  - {fn_csv}")
    print(f"  - {fp_csv}")

def main():
    if len(sys.argv) < 2:
        print("사용법: python analyze_fn_fp.py <결과_디렉토리> [데이터셋_디렉토리]")
        print("예: python analyze_fn_fp.py /data/jhmun0206/results/fraudgt/ellipticpp_hetero_cluster/hetero-cluster/config/0")
        print("    python analyze_fn_fp.py <결과_디렉토리> /data/jhmun0206/datasets/ellipticpp")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    dataset_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_fn_fp(result_dir, dataset_dir)

if __name__ == '__main__':
    main()

