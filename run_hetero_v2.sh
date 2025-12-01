#!/bin/bash
# 실제 이종 그래프 버전 학습 실행 스크립트
# 기존 결과 보존하면서 새로운 실험 진행

echo "=========================================="
echo "실제 이종 그래프 버전 학습 시작"
echo "=========================================="

# 프로젝트 루트로 이동
cd /data/jhmun0206/repos/FraudGT

# 기존 결과 백업 (선택사항)
# 기존 결과는 /data/jhmun0206/results/fraudgt/ellipticpp_multi_hetero_v1 에 그대로 보존됨

# 새로운 이종 그래프 버전으로 3번 학습
echo "3번 학습 시작 (seed 0, 1, 2)..."
python fraudGT/main.py --cfg configs/hetero-multi-v2.yaml --repeat 3 --gpu 0

echo "=========================================="
echo "학습 완료!"
echo "=========================================="
echo "기존 결과: /data/jhmun0206/results/fraudgt/ellipticpp_multi_hetero_v1/hetero-multi/"
echo "새 결과:   /data/jhmun0206/results/fraudgt/ellipticpp_multi_hetero_v2/hetero-multi-v2/"
echo "=========================================="

