# GitHub 업로드 가이드

이 문서는 이종 그래프 실험 코드를 GitHub에 업로드하는 방법을 안내합니다.

## 1. Git 저장소 초기화

```bash
cd /Users/jeonghwan/Desktop/Personal_projects/my_sftp_project
git init
```

## 2. 수정한 파일들 추가

### 필수 파일 (이종 그래프 실험 관련)

```bash
# 새로 생성한 데이터셋 클래스
git add fraudGT/datasets/ellipticpp_tx_wallet_pyg.py
git add fraudGT/datasets/__init__.py

# 설정 파일
git add fraudGT/config/dataset_config.py
git add configs/ellipticpp-txwallet-hgt.yaml

# 로더 수정
git add fraudGT/loader/master_loader.py

# 그래프 로더 수정
git add fraudGT/graphgym/loader.py

# HGT 모델 수정
git add fraudGT/graphgym/models/hgt.py

# README 및 문서
git add README.md
git add REPRODUCTION_GUIDE.md
git add HETERO_GRAPH_GUIDE.md
git add CHECKLIST_BEFORE_UPLOAD.md
```

### 선택적 파일 (분석/테스트 스크립트)

```bash
# 테스트 스크립트
git add test_dataset_loading.py
git add test_config.py
git add check_dataset_structure.py

# 분석 스크립트
git add analyze_fn_fp.py
git add analyze_model_performance.py
git add check_results.py

# 실행 스크립트
git add run_ellipticpp_txwallet.sh
git add run_ellipticpp.py

# 기타 문서
git add REVIEW_REPORT.md
```

### .gitignore 확인

```bash
# .gitignore에 이미 포함된 항목 확인
cat .gitignore
```

## 3. 첫 커밋 생성

```bash
git commit -m "Add heterogeneous graph support for Elliptic++ dataset

- Add EllipticPPPyG_TxWallet dataset class for tx+wallet heterogeneous graph
- Add target_ntype configuration support
- Modify loader and HGT model to support target node type classification
- Add configuration file for tx+wallet HGT training
- Add documentation and reproduction guide"
```

## 4. GitHub 저장소 생성 및 연결

### GitHub에서 새 저장소 생성
1. GitHub 웹사이트 접속
2. "New repository" 클릭
3. 저장소 이름 입력 (예: `fraudgt-ellipticpp-hetero`)
4. Public/Private 선택
5. "Create repository" 클릭

### 원격 저장소 연결

```bash
# GitHub 저장소 URL을 YOUR_USERNAME과 REPO_NAME으로 변경
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 또는 SSH 사용 시
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
```

## 5. 푸시

```bash
# 기본 브랜치를 main으로 설정 (GitHub 기본값)
git branch -M main

# 원격 저장소에 푸시
git push -u origin main
```

## 6. (선택) 추가 파일 업로드

나중에 더 많은 파일을 추가하려면:

```bash
# 모든 변경사항 확인
git status

# 특정 파일만 추가
git add <파일명>

# 또는 모든 변경사항 추가 (주의: .gitignore 제외)
git add .

# 커밋
git commit -m "Add additional files"

# 푸시
git push
```

## 7. (선택) .gitignore 업데이트

대용량 파일이나 민감한 정보는 제외:

```bash
# .gitignore에 추가할 항목 예시
echo "*.pt" >> .gitignore  # PyTorch 모델 파일
echo "*.pth" >> .gitignore
echo "results/" >> .gitignore  # 결과 디렉토리
echo "logs/" >> .gitignore  # 로그 디렉토리
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

## 주의사항

1. **대용량 파일**: `.pt` 파일(처리된 데이터셋)은 GitHub에 올리지 않는 것이 좋습니다. 대신 데이터셋 생성 방법을 문서화하세요.

2. **민감한 정보**: 
   - 절대 경로 (`/data/jhmun0206/...`)는 상대 경로나 환경 변수로 변경
   - API 키나 비밀번호는 절대 커밋하지 않음

3. **FraudGT 원본 코드**: 
   - FraudGT의 원본 코드는 라이선스를 확인하고 필요시 LICENSE 파일 추가

## 빠른 업로드 (한 번에)

```bash
# 1. 저장소 초기화
git init

# 2. 필수 파일만 추가
git add fraudGT/datasets/ellipticpp_tx_wallet_pyg.py \
        fraudGT/datasets/__init__.py \
        fraudGT/config/dataset_config.py \
        fraudGT/loader/master_loader.py \
        fraudGT/graphgym/loader.py \
        fraudGT/graphgym/models/hgt.py \
        configs/ellipticpp-txwallet-hgt.yaml \
        README.md REPRODUCTION_GUIDE.md HETERO_GRAPH_GUIDE.md

# 3. 커밋
git commit -m "Add heterogeneous graph support for Elliptic++"

# 4. GitHub 저장소 생성 후
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

