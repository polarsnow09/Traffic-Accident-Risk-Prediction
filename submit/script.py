import os, argparse, joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier 
from sklearn.calibration import CalibratedClassifierCV 
import sys

# =======================
# 학습 때 사용한 전처리 (함수 이름: preprocess_data)
# =======================
def preprocess_data(df: pd.DataFrame, train_medians: pd.Series) -> pd.DataFrame:
    
    # 1. 메타데이터 및 불필요 컬럼 제거 (학습 시와 동일하게 Age, 메타데이터 제거)
    test_ids = df['Test_id'].copy()
    
    drop_cols = ['Test', 'PrimaryKey', 'TestDate', 'Age', 'Label'] 
    
    X_test_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Test_id만 남기고 모두 수치형으로 변환 시도
    X_test_imputed = X_test_raw.drop(columns=['Test_id'], errors='ignore').apply(pd.to_numeric, errors='coerce')
    
    # 2. 결측치 대체 
    X_test = X_test_imputed.fillna(train_medians)
    
    return X_test, test_ids


def align_final_features(X_df, model):
    # CalibratedClassifierCV의 내부 estimator(XGB)의 feature_name_을 얻습니다.
    base_model = getattr(model, 'base_estimator', getattr(model, 'estimator', None))
    feat_names = list(getattr(base_model, "feature_names_in_", X_df.columns))
    
    X = X_df.copy()
    
    # 누락 피처 0으로 채움
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    
    # 초과 피처 드롭 + 순서 일치 (학습 시와 동일한 피처만, 순서도 동일하게)
    X = X[[c for c in feat_names if c in X.columns]]
    
    return X


# =======================
# main 함수: 모델 로드 및 예측 로직
# =======================
def main():
    # ---- 경로 변수 ----
    TEST_DIR    = "data"
    MODEL_DIR   = "model"
    OUT_DIR     = "output"
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH    = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 및 통계량 로드 ----
    print("Load models and medians...")
    try:
        model_A = joblib.load(os.path.join(MODEL_DIR, "Target_A_model.pkl"))
        model_B = joblib.load(os.path.join(MODEL_DIR, "Target_B_model.pkl"))
        medians_A = joblib.load(os.path.join(MODEL_DIR, "train_medians_A.pkl"))
        medians_B = joblib.load(os.path.join(MODEL_DIR, "train_medians_B.pkl"))
    except Exception as e:
        print(f"🚨 모델 또는 통계량 로드 오류: {e}", file=sys.stderr)
        sys.exit(1)
    print(" OK.")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    meta = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    # 예시 코드와 동일하게 경로 지정
    Araw = pd.read_csv(os.path.join(TEST_DIR, "test", "A.csv"))
    Braw = pd.read_csv(os.path.join(TEST_DIR, "test", "B.csv"))

    # ---- 매핑 ----
    A_df = meta.loc[meta["Test"] == "A", ["Test_id", "Test"]].merge(Araw, on="Test_id", how="left")
    B_df = meta.loc[meta["Test"] == "B", ["Test_id", "Test"]].merge(Braw, on="Test_id", how="left")

    # ---- 전처리 및 피처 정렬 ----
    
    # 1. 전처리 실행 (안전한 함수 호출)
    XA_df, test_ids_A = preprocess_data(A_df, medians_A)
    XB_df, test_ids_B = preprocess_data(B_df, medians_B)
    
    # 2. 피처 정렬
    XA = align_final_features(XA_df, model_A)
    XB = align_final_features(XB_df, model_B)
    print(f" aligned: XA={XA.shape}, XB={XB.shape}")
    
    # ---- 예측 ----
    print("Inference Model...")
    predA = model_A.predict_proba(XA)[:,1] if len(XA) else np.array([])
    predB = model_B.predict_proba(XB)[:,1] if len(XB) else np.array([])

    # 1. Target A/B의 예측 확률을 단일 DataFrame으로 만듭니다.
    probs_A = pd.DataFrame({"Test_id": test_ids_A, "prob_A": predA})
    probs_B = pd.DataFrame({"Test_id": test_ids_B, "prob_B": predB})
    
    # 2. meta 정보(test.csv)와 병합하여 모든 Test_id에 대한 A/B 확률 컬럼을 만듬
    result_df = meta.merge(probs_A, on="Test_id", how="left")
    result_df = result_df.merge(probs_B, on="Test_id", how="left")
    
    # 3. 우리의 최종 로직 (A/B 확률의 NaN 무시 평균)을 'prob' 컬럼에 계산
    result_df['prob'] = result_df[['prob_A', 'prob_B']].mean(axis=1, skipna=True)
    
    # ---- sample_submission 기반 결과 생성 (예시와 동일한 방식) ----
    os.makedirs(OUT_DIR, exist_ok=True)
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    
    # 4. sample의 Test_id 순서에 맞추어 prob 병합
    out = sample.merge(result_df[["Test_id", "prob"]], on="Test_id", how="left")
    
    # 5. 기존 'Label' 컬럼을 'prob' 값으로 덮어쓰고, 중간에 생긴 'prob' 컬럼을 제거
    out["Label"] = out["prob"].astype(float).fillna(0.0)
    out = out.drop(columns=["prob"])

    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved: {OUT_PATH} (rows={len(out)})")

if __name__ == "__main__":
    main()