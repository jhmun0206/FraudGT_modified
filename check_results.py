#!/usr/bin/env python
"""
학습 결과 확인 스크립트

사용법:
    python check_results.py /data/jhmun0206/results/fraudgt/ellipticpp_hetero_cluster/hetero-cluster/config/0
    python check_results.py <결과_디렉토리>
"""

import sys
import os
import json
from pathlib import Path

def find_file_in_paths(filename, search_paths):
    """여러 경로에서 파일 찾기"""
    for path in search_paths:
        filepath = Path(path) / filename
        if filepath.exists():
            return filepath
    return None

def check_results(result_dir):
    """결과 디렉토리 확인"""
    result_dir = Path(result_dir)
    
    print("=" * 80)
    print("학습 결과 확인")
    print("=" * 80)
    print(f"결과 디렉토리: {result_dir}")
    print()
    
    # 파일 검색 경로 설정
    # 1. 개별 run 디렉토리 (예: config/0/)
    # 2. agg/test/ 디렉토리 (예: config/agg/test/)
    # 3. agg/train/ 디렉토리 (예: config/agg/train/)
    search_paths = [
        result_dir,  # 개별 run 디렉토리
        result_dir.parent / 'agg' / 'test',  # agg/test
        result_dir.parent / 'agg' / 'train',  # agg/train
        result_dir / 'test',  # run/test (혹시 있을 경우)
        result_dir / 'train',  # run/train (혹시 있을 경우)
    ]
    
    # 1. 파일 존재 확인
    files_to_check = {
        'best.json': ['agg/test', 'agg/train', 'test', 'train', '.'],
        'stats.json': ['agg/test', 'agg/train', 'test', 'train', '.'],
        'test_predictions.npy': ['.'],  # 개별 run에만 있음
        'test_labels.npy': ['.'],  # 개별 run에만 있음
        'logging.log': ['.'],  # 개별 run에만 있음
    }
    
    print("[1] 파일 존재 확인:")
    found_files = {}
    for filename, locations in files_to_check.items():
        found = False
        for loc in locations:
            if loc == '.':
                search_path = result_dir
            else:
                search_path = result_dir.parent / loc
            filepath = search_path / filename
            if filepath.exists():
                found_files[filename] = filepath
                size_bytes = filepath.stat().st_size
                if size_bytes > 1024 * 1024:
                    size = f" ({size_bytes / (1024*1024):.2f} MB)"
                elif size_bytes > 1024:
                    size = f" ({size_bytes / 1024:.2f} KB)"
                else:
                    size = f" ({size_bytes} bytes)"
                print(f"  ✅ {filename}{size} (위치: {filepath.parent.name})")
                found = True
                break
        if not found:
            print(f"  ❌ {filename}")
    
    # 2. best.json 확인
    print("\n[2] Best 결과:")
    best_file = found_files.get('best.json')
    if best_file and best_file.exists():
        with open(best_file, 'r') as f:
            best = json.load(f)
        print(f"  Epoch: {best.get('epoch', 'N/A')}")
        print(f"  Accuracy: {best.get('accuracy', 'N/A')}")
        print(f"  Precision: {best.get('precision', 'N/A')}")
        print(f"  Recall: {best.get('recall', 'N/A')}")
        print(f"  F1: {best.get('f1', 'N/A')}")
        print(f"  Micro-F1: {best.get('micro-f1', 'N/A')}")
    else:
        print("  best.json 파일이 없습니다.")
    
    # 3. test_predictions.npy 확인
    print("\n[3] Test 예측 결과:")
    pred_file = found_files.get('test_predictions.npy') or (result_dir / 'test_predictions.npy')
    label_file = found_files.get('test_labels.npy') or (result_dir / 'test_labels.npy')
    
    if pred_file.exists() and label_file.exists():
        import numpy as np
        predictions = np.load(pred_file)
        labels = np.load(label_file)
        
        print(f"  예측 shape: {predictions.shape}")
        print(f"  레이블 shape: {labels.shape}")
        
        if len(predictions.shape) == 2:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        # Confusion matrix 계산
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, pred_classes)
        print(f"\n  Confusion Matrix:")
        print(f"    {cm}")
        
        # FN, FP 계산
        if cm.shape == (3, 3):  # 3-class
            # FN: 실제 fraud(2)인데 다른 클래스로 예측
            fn = cm[2, 0] + cm[2, 1]  # 실제 2인데 0 또는 1로 예측
            # FP: 실제 normal(0) 또는 unknown(1)인데 fraud(2)로 예측
            fp = cm[0, 2] + cm[1, 2]  # 실제 0 또는 1인데 2로 예측
            print(f"\n  FN (False Negative): {fn}개")
            print(f"  FP (False Positive): {fp}개")
    else:
        print("  test_predictions.npy 또는 test_labels.npy 파일이 없습니다.")
    
    # 4. stats.json 요약
    print("\n[4] 학습 통계 요약:")
    stats_file = found_files.get('stats.json')
    if stats_file and stats_file.exists():
        epochs = []
        with open(stats_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        stats = json.loads(line)
                        epochs.append(stats)
                    except:
                        pass
        
        if epochs:
            print(f"  총 에폭 수: {len(epochs)}")
            if epochs:
                last = epochs[-1]
                print(f"  마지막 에폭: {last.get('epoch', 'N/A')}")
                print(f"  마지막 Loss: {last.get('loss', 'N/A')}")
                print(f"  마지막 Accuracy: {last.get('accuracy', 'N/A')}")
                print(f"  마지막 F1: {last.get('f1', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("확인 완료!")
    print("=" * 80)

def main():
    if len(sys.argv) < 2:
        print("사용법: python check_results.py <결과_디렉토리>")
        print("예: python check_results.py /data/jhmun0206/results/fraudgt/ellipticpp_hetero_cluster/hetero-cluster/config/0")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    check_results(result_dir)

if __name__ == '__main__':
    main()

