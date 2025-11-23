#!/usr/bin/env python
"""
Config 파일 로드 및 출력 테스트 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import fraudGT  # noqa, register custom modules
from fraudGT.graphgym.cmd_args import parse_args
from fraudGT.graphgym.config import cfg, set_cfg, load_cfg, dump_cfg

def main():
    """Config 파일 로드 및 출력"""
    # 명령줄 인자 파싱 (--cfg만 필요)
    if len(sys.argv) < 2 or '--cfg' not in sys.argv:
        print("사용법: python test_config.py --cfg <config_file>")
        print("예: python test_config.py --cfg configs/ellipticpp-txwallet-hgt.yaml")
        sys.exit(1)
    
    # parse_args()를 직접 호출하면 에러가 날 수 있으므로 수동으로 파싱
    cfg_file = None
    for i, arg in enumerate(sys.argv):
        if arg == '--cfg' and i + 1 < len(sys.argv):
            cfg_file = sys.argv[i + 1]
            break
    
    if not cfg_file:
        print("ERROR: --cfg 옵션에 config 파일 경로를 지정하세요.")
        sys.exit(1)
    
    if not os.path.exists(cfg_file):
        print(f"ERROR: Config 파일을 찾을 수 없습니다: {cfg_file}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"Config 파일 로드: {cfg_file}")
    print("=" * 80)
    
    try:
        # Config 초기화
        set_cfg(cfg)
        
        # 간단한 args 객체 생성 (필수 필드만)
        class SimpleArgs:
            def __init__(self, cfg_file):
                self.cfg_file = cfg_file
                self.repeat = 1
                self.gpu = -1
                self.mark_done = False
                self.opts = []
        
        args = SimpleArgs(cfg_file)
        
        # Config 로드
        load_cfg(cfg, args)
        
        print("\n[INFO] Config 로드 성공!")
        print("\n[주요 설정]")
        print(f"  - Dataset: {cfg.dataset.name}")
        print(f"  - Dataset dir: {cfg.dataset.dir}")
        print(f"  - Format: {cfg.dataset.format}")
        print(f"  - Hetero: {getattr(cfg.dataset, 'hetero', False)}")
        print(f"  - Target ntype: {getattr(cfg.dataset, 'target_ntype', 'N/A')}")
        print(f"  - Model: {cfg.model.type}")
        print(f"  - GNN layer: {cfg.gnn.layer_type}")
        print(f"  - Output dir: {cfg.out_dir}")
        print(f"  - Device: {cfg.device}")
        print(f"  - Seed: {cfg.seed}")
        
        print("\n[전체 Config]")
        print(cfg)
        
        print("\n" + "=" * 80)
        print("Config 확인 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Config 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

