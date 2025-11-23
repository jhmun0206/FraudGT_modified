"""
온체인 이상거래 탐지 모델 성능 분석 스크립트

FN/FP 샘플 분석을 통해:
1. 모델이 왜 좋은 정확도를 보이는지
2. 놓친 데이터의 특징/공통점
3. 모델 성능 향상 방안 도출
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

def load_data(data_dir='.'):
    """데이터 로드"""
    data_dir = Path(data_dir)
    
    fn_df = pd.read_csv(data_dir / 'fn_samples_details.csv')
    fp_df = pd.read_csv(data_dir / 'fp_samples_details.csv')
    
    # 전체 예측 결과도 로드 (있는 경우)
    try:
        predictions = np.load(data_dir / 'test_predictions.npy')
        labels = np.load(data_dir / 'test_labels.npy')
        pred_classes = np.argmax(predictions, axis=1)
    except:
        predictions = None
        labels = None
        pred_classes = None
    
    return fn_df, fp_df, predictions, labels, pred_classes

def analyze_basic_stats(fn_df, fp_df):
    """기본 통계 분석"""
    print("=" * 80)
    print("1. 기본 통계 분석")
    print("=" * 80)
    
    print(f"\nFN (False Negative) - 놓친 Fraud:")
    print(f"  총 개수: {len(fn_df)}")
    print(f"  원본 클래스 분포:")
    print(fn_df['original_class'].value_counts().sort_index())
    
    print(f"\nFP (False Positive) - 잘못 의심:")
    print(f"  총 개수: {len(fp_df)}")
    print(f"  원본 클래스 분포:")
    print(fp_df['original_class'].value_counts().sort_index())
    
    print(f"\n예측 클래스 분포:")
    print(f"  FN 예측: {fn_df['predicted_label'].value_counts().sort_index()}")
    print(f"  FP 예측: {fp_df['predicted_label'].value_counts().sort_index()}")

def analyze_features(fn_df, fp_df):
    """특징 분석"""
    print("\n" + "=" * 80)
    print("2. 특징 분석")
    print("=" * 80)
    
    # 숫자형 특징만 선택
    numeric_cols = fn_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['test_index', 'node_index', 'tx_id', 'true_label', 'predicted_label']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if len(numeric_cols) == 0:
        print("숫자형 특징을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(numeric_cols)}개의 숫자형 특징 발견")
    
    # FN과 FP의 특징 평균 비교
    fn_means = fn_df[numeric_cols].mean()
    fp_means = fp_df[numeric_cols].mean()
    
    # 차이가 큰 특징 찾기
    diff = (fn_means - fp_means).abs().sort_values(ascending=False)
    
    print(f"\nFN과 FP 간 차이가 큰 특징 Top 10:")
    for feat, val in diff.head(10).items():
        print(f"  {feat}:")
        print(f"    FN 평균: {fn_means[feat]:.4f}")
        print(f"    FP 평균: {fp_means[feat]:.4f}")
        print(f"    차이: {val:.4f}")
    
    return diff.head(20)

def analyze_patterns(fn_df, fp_df):
    """패턴 분석"""
    print("\n" + "=" * 80)
    print("3. 패턴 분석")
    print("=" * 80)
    
    # 트랜잭션 ID 패턴 (예: 특정 범위에 집중되는지)
    print("\n트랜잭션 ID 분포:")
    print(f"  FN tx_id 범위: {fn_df['tx_id'].min()} ~ {fn_df['tx_id'].max()}")
    print(f"  FP tx_id 범위: {fp_df['tx_id'].min()} ~ {fp_df['tx_id'].max()}")
    
    # 노드 인덱스 분포 (그래프 상 위치)
    print(f"\n노드 인덱스 분포:")
    print(f"  FN 노드 범위: {fn_df['node_index'].min()} ~ {fn_df['node_index'].max()}")
    print(f"  FP 노드 범위: {fp_df['node_index'].min()} ~ {fp_df['node_index'].max()}")
    
    # 클러스터링 가능성 확인 (특정 범위에 집중되는지)
    fn_node_clusters = fn_df['node_index'].describe()
    fp_node_clusters = fp_df['node_index'].describe()
    
    print(f"\n노드 인덱스 통계:")
    print(f"  FN - 중앙값: {fn_node_clusters['50%']:.0f}, 표준편차: {fn_node_clusters['std']:.0f}")
    print(f"  FP - 중앙값: {fp_node_clusters['50%']:.0f}, 표준편차: {fp_node_clusters['std']:.0f}")

def visualize_comparison(fn_df, fp_df, top_features, output_dir='.'):
    """시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("4. 시각화 생성")
    print("=" * 80)
    
    # 숫자형 특징만 선택
    numeric_cols = fn_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['test_index', 'node_index', 'tx_id', 'true_label', 'predicted_label']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if len(numeric_cols) == 0:
        print("시각화할 특징이 없습니다.")
        return
    
    # Top 특징들의 분포 비교
    if top_features is not None and len(top_features) > 0:
        n_features = min(6, len(top_features))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feat in enumerate(top_features.head(n_features).index):
            ax = axes[i]
            
            # FN과 FP의 분포 비교
            ax.hist(fn_df[feat].dropna(), bins=30, alpha=0.5, label='FN (놓친 Fraud)', color='red')
            ax.hist(fp_df[feat].dropna(), bins=30, alpha=0.5, label='FP (잘못 의심)', color='blue')
            ax.set_xlabel(feat)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feat} 분포 비교')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  저장됨: {output_dir / 'feature_comparison.png'}")
        plt.close()
    
    # 클래스 분포 비교
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 원본 클래스 분포
    fn_class_dist = fn_df['original_class'].value_counts().sort_index()
    fp_class_dist = fp_df['original_class'].value_counts().sort_index()
    
    axes[0].bar(fn_class_dist.index - 0.2, fn_class_dist.values, width=0.4, label='FN', color='red', alpha=0.7)
    axes[0].bar(fp_class_dist.index + 0.2, fp_class_dist.values, width=0.4, label='FP', color='blue', alpha=0.7)
    axes[0].set_xlabel('Original Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('원본 클래스 분포')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 예측 클래스 분포
    fn_pred_dist = fn_df['predicted_label'].value_counts().sort_index()
    fp_pred_dist = fp_df['predicted_label'].value_counts().sort_index()
    
    axes[1].bar(fn_pred_dist.index - 0.2, fn_pred_dist.values, width=0.4, label='FN', color='red', alpha=0.7)
    axes[1].bar(fp_pred_dist.index + 0.2, fp_pred_dist.values, width=0.4, label='FP', color='blue', alpha=0.7)
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title('예측 클래스 분포')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  저장됨: {output_dir / 'class_distribution.png'}")
    plt.close()

def generate_insights(fn_df, fp_df, top_features):
    """인사이트 생성"""
    print("\n" + "=" * 80)
    print("5. 모델 성능 인사이트")
    print("=" * 80)
    
    insights = []
    
    # 1. FN 분석
    print("\n[FN (놓친 Fraud) 분석]")
    print("-" * 80)
    
    # FN의 특징
    numeric_cols = fn_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['test_index', 'node_index', 'tx_id', 'true_label', 'predicted_label']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if len(numeric_cols) > 0:
        fn_means = fn_df[numeric_cols].mean()
        print(f"FN 샘플의 평균 특징값 (주요 특징):")
        for feat in numeric_cols[:10]:
            print(f"  {feat}: {fn_means[feat]:.4f}")
    
    insights.append(f"FN: {len(fn_df)}개의 실제 fraud를 놓쳤습니다.")
    insights.append(f"  - 대부분 실제 클래스 {fn_df['original_class'].mode()[0]}에서 발생")
    insights.append(f"  - 모두 Normal(0)으로 잘못 예측됨")
    
    # 2. FP 분석
    print("\n[FP (잘못 의심) 분석]")
    print("-" * 80)
    
    if len(numeric_cols) > 0:
        fp_means = fp_df[numeric_cols].mean()
        print(f"FP 샘플의 평균 특징값 (주요 특징):")
        for feat in numeric_cols[:10]:
            print(f"  {feat}: {fp_means[feat]:.4f}")
    
    insights.append(f"\nFP: {len(fp_df)}개의 정상 거래를 잘못 의심했습니다.")
    insights.append(f"  - 대부분 실제 클래스 {fp_df['original_class'].mode()[0]}에서 발생")
    pred_dist = fp_df['predicted_label'].value_counts()
    insights.append(f"  - {pred_dist.idxmax()}({pred_dist.max()}개)로 잘못 예측")
    
    # 3. 개선 방안
    print("\n[모델 개선 방안]")
    print("-" * 80)
    
    if top_features is not None and len(top_features) > 0:
        print("차이가 큰 특징을 활용한 개선 방안:")
        for feat in top_features.head(5).index:
            fn_mean = fn_df[feat].mean()
            fp_mean = fp_df[feat].mean()
            print(f"  - {feat}: FN과 FP 간 평균 차이 {abs(fn_mean - fp_mean):.4f}")
            if fn_mean > fp_mean:
                print(f"    → FN이 더 높은 값을 가짐. 이 특징에 대한 임계값 조정 고려")
            else:
                print(f"    → FP가 더 높은 값을 가짐. 이 특징에 대한 임계값 조정 고려")
    
    insights.append("\n개선 제안:")
    insights.append("  1. FN과 FP 간 차이가 큰 특징에 가중치 부여")
    insights.append("  2. 클래스 불균형 고려한 손실 함수 조정")
    insights.append("  3. 그래프 구조 정보 활용 강화 (이웃 노드 레이블 고려)")
    
    print("\n".join(insights))
    
    return insights

def main():
    """메인 분석 함수"""
    print("=" * 80)
    print("온체인 이상거래 탐지 모델 성능 분석")
    print("=" * 80)
    
    # 데이터 로드
    data_dir = Path('.')
    fn_df, fp_df, predictions, labels, pred_classes = load_data(data_dir)
    
    # 분석 수행
    analyze_basic_stats(fn_df, fp_df)
    top_features = analyze_features(fn_df, fp_df)
    analyze_patterns(fn_df, fp_df)
    visualize_comparison(fn_df, fp_df, top_features, data_dir)
    insights = generate_insights(fn_df, fp_df, top_features)
    
    # 결과 저장
    output_file = data_dir / 'analysis_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("온체인 이상거래 탐지 모델 성능 분석 보고서\n")
        f.write("=" * 80 + "\n\n")
        f.write("\n".join(insights))
    
    print(f"\n분석 완료! 보고서 저장: {output_file}")
    print(f"시각화 파일: feature_comparison.png, class_distribution.png")

if __name__ == '__main__':
    main()

