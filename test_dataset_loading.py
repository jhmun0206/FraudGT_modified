#!/usr/bin/env python
"""
데이터셋 로딩 테스트 스크립트

이 스크립트는 데이터셋이 정상적으로 로드되는지 확인하기 위한 것입니다.
학습 전에 반드시 실행하여 데이터셋 로딩이 1분 이내에 완료되는지 확인하세요.

사용법:
    python test_dataset_loading.py --cfg configs/hetero-multi-v2.yaml
"""

import sys
import os
import time
import argparse

# 프로젝트 루트를 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from fraudGT.graphgym.config import cfg, set_cfg
    from fraudGT.loader.master_loader import load_dataset_master
except ImportError as e:
    print(f"[ERROR] 모듈 import 실패: {e}")
    print("프로젝트 루트 디렉토리에서 실행하세요.")
    sys.exit(1)


def test_dataset_loading(config_path=None):
    """데이터셋 로딩 테스트"""
    print("=" * 60)
    print("데이터셋 로딩 테스트 시작")
    print("=" * 60)
    
    # Config 로드
    if config_path:
        print(f"[INFO] Config 파일 로드: {config_path}")
        set_cfg(cfg)
        cfg.merge_from_file(config_path)
        cfg.freeze()
    else:
        print("[WARNING] Config 파일이 지정되지 않았습니다. 기본값을 사용합니다.")
        # 최소한의 설정
        if not hasattr(cfg, 'dataset'):
            print("[ERROR] Config가 제대로 로드되지 않았습니다.")
            return False
    
    dataset_dir = cfg.dataset.dir
    dataset_name = cfg.dataset.name
    hetero_version = getattr(cfg.dataset, 'hetero_version', None)
    
    print(f"[INFO] 데이터셋 디렉토리: {dataset_dir}")
    print(f"[INFO] 데이터셋 이름: {dataset_name}")
    print(f"[INFO] 이종 그래프 버전: {hetero_version}")
    print()
    
    # 타임아웃 설정 (5분)
    timeout_seconds = 300
    
    start_time = time.time()
    
    try:
        print("[INFO] 데이터셋 로딩 시작...")
        print(f"[INFO] 타임아웃: {timeout_seconds}초 ({timeout_seconds//60}분)")
        print()
        
        # 데이터셋 로드
        dataset = load_dataset_master(
            format=cfg.dataset.format,
            name=dataset_name,
            dataset_dir=dataset_dir
        )
        
        elapsed_time = time.time() - start_time
        
        print()
        print("=" * 60)
        print("데이터셋 로딩 성공!")
        print("=" * 60)
        print(f"[INFO] 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
        
        # 데이터셋 정보 출력
        if hasattr(dataset, 'data'):
            data = dataset.data if hasattr(dataset, 'data') else dataset[0]
            
            print()
            print("[INFO] 데이터셋 구조:")
            if hasattr(data, 'node_types'):
                # HeteroData
                print(f"  - 노드 타입: {data.node_types}")
                print(f"  - 엣지 타입: {data.edge_types}")
                for node_type in data.node_types:
                    if hasattr(data[node_type], 'x'):
                        print(f"  - {node_type} 노드 수: {data[node_type].x.size(0):,}")
                for edge_type in data.edge_types:
                    if hasattr(data[edge_type], 'edge_index'):
                        num_edges = data[edge_type].edge_index.size(1)
                        print(f"  - {edge_type} 엣지 수: {num_edges:,}")
            else:
                # Data
                print(f"  - 노드 수: {data.x.size(0):,}")
                if hasattr(data, 'edge_index'):
                    print(f"  - 엣지 수: {data.edge_index.size(1):,}")
        
        # 권장 사항
        print()
        if elapsed_time < 60:
            print("[SUCCESS] ✅ 데이터셋 로딩이 1분 이내에 완료되었습니다.")
            print("[INFO] 학습을 진행해도 좋습니다.")
        elif elapsed_time < 300:
            print("[WARNING] ⚠️  데이터셋 로딩이 1-5분 사이에 완료되었습니다.")
            print("[INFO] 학습은 가능하지만, 샘플링 비율을 더 줄이는 것을 고려하세요.")
        else:
            print("[ERROR] ❌ 데이터셋 로딩이 5분 이상 걸렸습니다.")
            print("[INFO] 샘플링 비율과 엣지 수 상한을 더 줄여야 합니다.")
            print("[INFO] config 파일의 다음 설정을 확인하세요:")
            print("  - dataset.hetero_v2_sample_ratio (기본값: 0.02)")
            print("  - dataset.hetero_v2_max_tx_samples (기본값: 20000)")
            print("  - dataset.hetero_v2_max_tx_to_addr_edges (기본값: 50000)")
            return False
        
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print()
        print("=" * 60)
        print("데이터셋 로딩 실패!")
        print("=" * 60)
        print(f"[ERROR] 소요 시간: {elapsed_time:.2f}초")
        print(f"[ERROR] 에러 메시지: {e}")
        import traceback
        print()
        print("[ERROR] 상세 스택 트레이스:")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='데이터셋 로딩 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
예제:
  python test_dataset_loading.py --cfg configs/hetero-multi-v2.yaml
  python test_dataset_loading.py  # 기본값: configs/hetero-multi-v2.yaml 사용
        '''
    )
    parser.add_argument('--cfg', type=str, 
                       default='configs/hetero-multi-v2.yaml',
                       help='Config 파일 경로 (기본값: configs/hetero-multi-v2.yaml)')
    parser.add_argument('--dataset-dir', type=str, help='데이터셋 디렉토리 (선택사항, config에서 오버라이드)')
    
    args = parser.parse_args()
    
    # Config 파일 존재 확인
    if not os.path.exists(args.cfg):
        print(f"[ERROR] Config 파일을 찾을 수 없습니다: {args.cfg}")
        print(f"[INFO] 현재 디렉토리: {os.getcwd()}")
        print(f"[INFO] 사용 가능한 config 파일:")
        config_dir = 'configs'
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.endswith('.yaml') or f.endswith('.yml'):
                    print(f"  - {os.path.join(config_dir, f)}")
        sys.exit(1)
    
    success = test_dataset_loading(args.cfg)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

