"""
원본 데이터셋 구조 확인 스크립트
실제 이종 그래프를 만들기 위해 원본 데이터의 구조를 확인
"""

import pandas as pd
import os

def check_dataset_structure(dataset_dir):
    """원본 데이터셋 구조 확인"""
    print("=" * 80)
    print("원본 데이터셋 구조 확인")
    print("=" * 80)
    
    # 1. 특징 파일 확인
    feats_file = os.path.join(dataset_dir, "txs_features.csv")
    if os.path.exists(feats_file):
        print(f"\n[1] 특징 파일: {feats_file}")
        feats = pd.read_csv(feats_file, nrows=5)  # 처음 5줄만
        print(f"   컬럼 수: {len(feats.columns)}")
        print(f"   컬럼 목록:")
        for i, col in enumerate(feats.columns[:20]):  # 처음 20개만
            print(f"     {i+1}. {col}")
        if len(feats.columns) > 20:
            print(f"     ... (총 {len(feats.columns)}개 컬럼)")
        
        # 주소 관련 컬럼 찾기
        address_keywords = ['address', 'addr', 'input', 'output', 'from', 'to', 
                           'sender', 'receiver', 'wallet', 'account']
        address_cols = []
        for col in feats.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in address_keywords):
                address_cols.append(col)
        
        if address_cols:
            print(f"\n   주소 관련 컬럼 발견: {address_cols}")
        else:
            print(f"\n   주소 관련 컬럼 없음 (트랜잭션 간 연결만 사용)")
    else:
        print(f"[ERROR] 특징 파일 없음: {feats_file}")
    
    # 2. 엣지 파일 확인
    edges_file = os.path.join(dataset_dir, "txs_edgelist.csv")
    if os.path.exists(edges_file):
        print(f"\n[2] 엣지 파일: {edges_file}")
        edges = pd.read_csv(edges_file, nrows=10)
        print(f"   컬럼: {list(edges.columns)}")
        print(f"   샘플 데이터:")
        print(edges.head())
        
        # 전체 엣지 수 확인
        total_edges = sum(1 for _ in open(edges_file)) - 1  # 헤더 제외
        print(f"   총 엣지 수: {total_edges:,}")
    else:
        print(f"[ERROR] 엣지 파일 없음: {edges_file}")
    
    # 3. 레이블 파일 확인
    labels_file = os.path.join(dataset_dir, "txs_classes.csv")
    if os.path.exists(labels_file):
        print(f"\n[3] 레이블 파일: {labels_file}")
        labels = pd.read_csv(labels_file, nrows=10)
        print(f"   컬럼: {list(labels.columns)}")
        print(f"   샘플 데이터:")
        print(labels.head())
        
        # 클래스 분포 확인
        label_col = None
        for col in labels.columns:
            if 'class' in col.lower() or 'label' in col.lower() or col.lower() == 'y':
                label_col = col
                break
        
        if label_col:
            full_labels = pd.read_csv(labels_file)
            print(f"\n   클래스 분포 ({label_col}):")
            print(full_labels[label_col].value_counts().sort_index())
    else:
        print(f"[ERROR] 레이블 파일 없음: {labels_file}")
    
    print("\n" + "=" * 80)
    print("이종 그래프 구성 제안")
    print("=" * 80)
    
    if address_cols:
        print("\n✅ 주소 정보 발견!")
        print("   → 실제 이종 그래프 구성 가능:")
        print("     - 노드 타입 1: 'tx' (트랜잭션)")
        print("     - 노드 타입 2: 'address' (주소/지갑)")
        print("     - 엣지: ('tx', 'to', 'address'), ('address', 'from', 'tx')")
    else:
        print("\n⚠️  주소 정보 없음")
        print("   → 대안 방법:")
        print("     1. 트랜잭션 특징에서 주소 정보 추출 시도")
        print("     2. 트랜잭션을 시간/블록 단위로 그룹화하여 블록 노드 생성")
        print("     3. 트랜잭션 클러스터링으로 메타 노드 생성")
        print("     4. 현재 구현: 트랜잭션 기반 주소 생성 (시뮬레이션)")

if __name__ == '__main__':
    dataset_dir = '/data/jhmun0206/datasets/ellipticpp'
    check_dataset_structure(dataset_dir)

