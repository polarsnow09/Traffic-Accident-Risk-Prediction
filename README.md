# 교통사고 위험 예측 AI 🚗⚠️

> 운수종사자 인지 특성 기반 교통사고 위험군 확률 예측

[![Score](https://img.shields.io/badge/Score-0.1908-blue)](https://github.com/polarsnow09/Traffic-Accident-Risk-Prediction)
[![Rank](https://img.shields.io/badge/Rank-207%2F437%20(48%25)-green)](https://github.com/polarsnow09/Traffic-Accident-Risk-Prediction)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)

## 🏆 주요 성과

- **Public Score**: 0.1908 (상위 48%, 207/437)
- **평가 지표**: 복합 지표 (AUC + Brier Score + ECE)
- **완전 자동화 파이프라인**: 학습 → 튜닝 → 보정 → 배포
- **배포 패키징**: pkl + script.py 형태로 실무 배포 구조 구현

## 🎯 프로젝트 개요

운수종사자 자격검사(신규/유지) 과정에서 수집된 **인지·반응 검사 데이터**를 활용하여,  
검사 결과 기준 **교통사고 위험군에 속할 확률**을 예측하는 AI 모델 개발

### 비즈니스 가치
```
✅ 운수종사자 사전 스크리닝
✅ 교통사고 예방 시스템
✅ 데이터 기반 자격 관리
```

## 📊 핵심 결과

### 평가 지표
```
Score = 0.5 × (1−AUC) + 0.25 × Brier + 0.25 × ECE

→ 3개 지표 동시 최적화 필요
→ 단순 분류 정확도가 아닌 "확률 보정" 중요
```

### 개선 과정
| Version | Score | 개선 | 주요 변경 |
|---------|-------|------|-----------|
| Version 1 | 0.1939 | - | Baseline (XGBoost) |
| Version 2 | 0.1933 | -0.6점 | RandomizedSearchCV (n=30) |
| Version 5 | 0.1916 | -1.7점 | GridSearchCV |
| **Version 8** | **0.1908** | **-0.8점** | **Median + CV=5 + Optuna** 🥇 |

**총 개선**: -1.6% (0.0031점)

## 🛠️ 기술 스택

### 모델 아키텍처
```
XGBoost → Optuna 튜닝 → Isotonic Calibration → 확률 예측
```

**핵심 기술**
- **모델**: XGBoost Classifier
- **튜닝**: Optuna (TPE Sampler, 40 trials, 20분)
- **보정**: CalibratedClassifierCV (isotonic, cv=5)
- **피처**: 자동 생성 통계 피처 55개
- **배포**: pkl + script.py 패키징

### 라이브러리
```python
pandas>=1.0.0
numpy>=1.18.0
joblib>=1.0.0
xgboost>=1.5.0
scikit-learn>=1.0.0
optuna>=3.0.0
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 모델 학습 (model_save.py)
```bash
python model_save.py
```

**출력:**
```
model/
├── Target_A_model.pkl           # A 자격 모델
├── Target_B_model.pkl           # B 자격 모델
├── train_medians_A.pkl          # A 통계량
└── train_medians_B.pkl          # B 통계량
```

### 3. 추론 및 제출 (script.py)
```bash
python script.py
```

**출력:**
```
output/
└── submission.csv               # 제출 파일
```

## 🎓 기술적 특징

### 1. 배포 중심 설계 ⭐⭐⭐

**까다로운 제출 조건:**
```
요구사항:
- model.pkl: 학습된 모델 저장
- script.py: 추론 코드 (자동 실행)
- requirements.txt: 의존성 관리
- zip 패키징 후 제출
- 평가 서버에서 자동 실행 (30분 제한)
```

**해결 전략:**

#### 문제 1: 피처 불일치
```python
def align_final_features(X_df, model):
    """학습 시 피처와 추론 시 피처를 정렬"""
    # Calibrated 모델에서 base_estimator의 피처 추출
    base_model = getattr(model, 'base_estimator', ...)
    feat_names = getattr(base_model, "feature_names_in_", ...)
    
    # 누락 피처 → 0으로 채움
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    
    # 순서 일치 (학습 시와 동일한 순서)
    X = X[[c for c in feat_names if c in X.columns]]
    return X
```

#### 문제 2: A/B 자격 분리
```python
# 별도 모델 2개 학습
model_A = train_model(X_train_A, Y_train_A)
model_B = train_model(X_train_B, Y_train_B)

# 별도 통계량 저장 (결측치 처리용)
joblib.dump(medians_A, 'train_medians_A.pkl')
joblib.dump(medians_B, 'train_medians_B.pkl')

# 추론 시 각각 예측 후 평균
prob = (prob_A + prob_B) / 2  # NaN 제외 평균
```

#### 문제 3: 재현성 보장
```python
# 학습 시: 통계량 저장
train_medians = X_train_imputed.median()
joblib.dump(train_medians, 'train_medians_A.pkl')

# 추론 시: 동일 통계량 사용
X_test = X_test_imputed.fillna(train_medians)

→ Data Leakage 0%, 재현성 100%
```

---

### 2. 고급 피처 엔지니어링 ⭐⭐

**자동 생성 통계 피처 55개:**

```python
def create_advanced_features(df):
    """도메인 무관 통계 피처 자동 생성"""
    
    # 기본 통계량 (6개)
    - mean, std, min, max, range, CV(변동계수)
    
    # 분포 특성 (2개)
    - skewness, kurtosis
    
    # 분위수 기반 (3개)
    - top10_ratio, bottom10_ratio, top25_ratio
    
    # 결측치 패턴 (3개)
    - missing_count, missing_ratio, complete_ratio
    
    # Z-score 변환 (2개)
    - z_mean, z_std
    
    # 비율 (1개)
    - mean_std_ratio
    
    # + 더 많은 파생 피처...
    
    return df  # 총 55개 피처 추가
```

**장점:**
- ✅ 도메인 지식 불필요
- ✅ 완전 자동화 가능
- ✅ 벡터화 계산 (빠름)
- ✅ 다른 프로젝트 재사용 가능

---

### 3. Optuna 하이퍼파라미터 튜닝 ⭐

```python
def tune_with_optuna(X_train, Y_train):
    """40 trials, 20분 제한 자동 튜닝"""
    
    # 튜닝 파라미터 (9개)
    params = {
        'n_estimators': (1000, 4000),
        'learning_rate': (0.005, 0.05, log),
        'max_depth': (3, 10),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'min_child_weight': (1, 10),
        'gamma': (0, 5),
        'reg_lambda': (0.01, 10, log),
        'reg_alpha': (0, 5)
    }
    
    study.optimize(objective, n_trials=40, timeout=1200)
    return study.best_params
```

---

### 4. Isotonic Calibration ⭐⭐

```python
# XGBoost 학습 후 확률 보정
calibrated_model = CalibratedClassifierCV(
    xgb_clf, 
    method='isotonic',  # 단조 변환
    cv=5                # 5-Fold CV
)
calibrated_model.fit(X_train, Y_train)
```

**왜 중요한가?**
```
평가 지표가 복합 지표:
- AUC: 분류 성능
- Brier Score: 확률 정확도
- ECE: 보정 오차

→ Calibration으로 Brier + ECE 최적화
→ "신뢰할 수 있는 확률" 제공
```

## 📂 프로젝트 구조

```
Traffic-Accident-Risk-Prediction/
├── model_save.py              # 학습 스크립트
├── data/                      # 데이터 
│   ├── train.csv
│   ├── test.csv
│   ├── train/
│   │   ├── A.csv
│   │   └── B.csv
│   └── test/
│       ├── A.csv
│       └── B.csv
└── submit/                    # 제출 파일 (zip 형식)
    ├── model/                 # 학습된 모델(4개 pkl)
    │   ├── Target_A_model.pkl
    │   ├── Target_B_model.pkl
    │   ├── train_medians_A.pkl
    │   └── train_medians_B.pkl
    ├── script.py                  # 추론 스크립트
    └── requirements.txt           # 필요한 패키지 및 버전

```

## 💡 핵심 학습

### 1. 실무 배포 경험
```
✅ 모델 직렬화 (pickle)
✅ 추론 스크립트 설계
✅ 피처 정렬 (align_final_features)
✅ 의존성 관리 (requirements.txt)
✅ 재현 가능성 보장 (통계량 저장)

→ MLOps 기초 경험
→ 실무 배포 형태 구현
```

### 2. Calibration의 중요성
```
단순 분류 정확도 (Accuracy) X
확률 보정 (Calibration) O

→ Brier Score 최적화
→ ECE 최적화
→ 신뢰할 수 있는 확률 예측
```

### 3. 자동화 파이프라인
```
수작업 최소화:
✅ 55개 피처 자동 생성
✅ Optuna 자동 튜닝 (40 trials)
✅ A/B 자동 분리 처리
✅ Early Stopping

→ 효율적 개발
→ 재현 가능성
```

### 4. 복합 지표 대응
```
단일 지표 최적화는 쉬움
복합 지표는 균형 필요

AUC + Brier + ECE 동시 최적화:
→ XGBoost (AUC 우수)
→ Calibration (Brier + ECE 우수)
→ 균형잡힌 모델
```

## 🔍 회고

### 잘한 점
- ✅ 까다로운 제출 조건 극복
- ✅ 완전 자동화 파이프라인 구축
- ✅ Calibration 적용으로 복합 지표 최적화
- ✅ 재현 가능한 배포 패키징

### 아쉬운 점
- ⚠️ 초기 EDA 부족
- ⚠️ 실험 과정 문서화 미흡 (프롬프트 로그 없음)
- ⚠️ 앙상블 기법 미적용

### 다음에는
- [ ] 더 철저한 데이터 분석 (상관관계, 분포)
- [ ] 실험 과정 체계적 문서화
- [ ] Stacking/Blending 앙상블 시도
- [ ] Feature Selection 전략 적용

## 📝 상세 문서

프로젝트의 전체 과정, 기술적 구현, 배포 전략은 아래 Notion 페이지에서 확인하세요:

🔗 **[프로젝트 포트폴리오 (Notion)](#)** ← 내일 작성 후 링크 추가!

- 📊 데이터 분석 및 EDA
- 🛠️ 피처 엔지니어링 상세
- 🎯 모델링 전략
- 📦 배포 패키징 과정 (핵심!)
- 💡 회고 및 핵심 학습

## 📞 연락처

- **이메일**: polarsnow09@gmail.com
- **GitHub**: [polarsnow09](https://github.com/polarsnow09)
- **LinkedIn**: [추가 예정]

---

**"모델 성능도 중요하지만, 실제 배포 가능한 형태로 패키징하는 능력이 실무에서 더욱 중요하다"**

이 프로젝트를 통해 **실무 MLOps의 기초**를 경험했습니다. 💪