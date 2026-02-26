import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------
# 0. 경로 설정 및 디렉토리 준비
# --------------------------------------------------------------------------
TRAIN_INFO_PATH = r'C:\Users\user\Desktop\데이터 분석\운수종사자_교통사고_위험예측\data\train.csv'
TRAIN_A_PATH = r'C:\Users\user\Desktop\데이터 분석\운수종사자_교통사고_위험예측\data\train\A.csv'
TRAIN_B_PATH = r'C:\Users\user\Desktop\데이터 분석\운수종사자_교통사고_위험예측\data\train\B.csv'

MODEL_DIR = 'model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"✅ 디렉토리 '{MODEL_DIR}/' 생성 완료.")

# --------------------------------------------------------------------------
# A. 데이터 로드 및 전처리 유틸리티
# --------------------------------------------------------------------------
def create_advanced_features(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    original_count = len(df.columns)
    
    print(f"\nLLM 만능 피처 엔지니어링 시작 → 숫자형 컬럼 {len(num_cols)}개 발견")
    
    arr = df[num_cols].values  # (n_samples, n_features) 넘파이 배열
    
    df['num_mean']   = np.nanmean(arr, axis=1)
    df['num_std']    = np.nanstd(arr, axis=1)
    df['num_min']    = np.nanmin(arr, axis=1)
    df['num_max']    = np.nanmax(arr, axis=1)
    df['num_range']  = df['num_max'] - df['num_min']
    df['num_cv']     = df['num_std'] / (np.abs(df['num_mean']) + 1e-8)
    
    # skew/kurt
    df['num_skew'] = df[num_cols].apply(lambda x: pd.Series(x).skew(), axis=1).fillna(0)
    df['num_kurt'] = df[num_cols].apply(lambda x: pd.Series(x).kurt(), axis=1).fillna(0)
    
    # 상하위 비율
    q90 = np.nanpercentile(arr, 90, axis=1)
    q10 = np.nanpercentile(arr, 10, axis=1)
    q75 = np.nanpercentile(arr, 75, axis=1)
    
    df['num_top10_ratio']    = (arr >= q90[:, None]).sum(axis=1) / len(num_cols)
    df['num_bottom10_ratio'] = (arr <= q10[:, None]).sum(axis=1) / len(num_cols)
    df['num_top25_ratio']    = (arr >= q75[:, None]).sum(axis=1) / len(num_cols)
    
    # 결측치
    df['missing_count']  = np.isnan(arr).sum(axis=1)
    df['missing_ratio']  = df['missing_count'] / len(num_cols)
    df['complete_ratio'] = 1 - df['missing_ratio']
    
    # Z-score (벡터화 계산)
    mean_vec = np.nanmean(arr, axis=1, keepdims=True)
    std_vec  = np.nanstd(arr, axis=1, keepdims=True)
    z_arr = (arr - mean_vec) / (std_vec + 1e-8)
    df['z_mean'] = np.nanmean(z_arr, axis=1)
    df['z_std']  = np.nanstd(z_arr, axis=1)
    
    df['mean_std_ratio'] = df['num_mean'] / (df['num_std'] + 1e-8)
    
    new_features = len(df.columns) - original_count
    print(f"LLM 만능 피처 엔지니어링 성공 → {new_features}개 추가 생성!")
    print(f"최종 피처 수: {len(df.columns)}개")
    
    return df

def load_and_preprocess_train_data(train_df, train_info_df, target_test):
    """훈련 데이터를 로드, 전처리하고 중앙값 통계량을 반환합니다."""
    
    # 1. 타겟 변수 병합
    y_info = train_info_df[train_info_df['Test'] == target_test][['Test_id', 'Label']].rename(columns={'Label': 'Target'})
    train_merged = pd.merge(train_df, y_info, on='Test_id', how='inner')
    
    Y_train = train_merged['Target']
    
    # 2. 불필요 컬럼 제거 (script.py의 전처리 로직과 일치해야 함)
    X_train_raw = train_merged.drop(columns=[
        'Test_id', 'Target', 'Test', 'PrimaryKey', 'TestDate', 'Age', 'Label' # 'Label'은 안전을 위해 추가
    ], errors='ignore')
    
    # 고급 피처 엔지니어링 (55개 피처)
    print(f"LLM 기반 고급 피처 엔지니어링 진행 중... ({target_test})")
    X_train_raw = create_advanced_features(X_train_raw)
    print(f"피처 수: {len(X_train_raw.columns)}개.")

    # 3. 수치형 변환 및 결측치 확인
    X_train_imputed = X_train_raw.apply(pd.to_numeric, errors='coerce')
    
    # 4. 중앙값 계산 및 결측치 대체
    train_medians = X_train_imputed.median()
    X_train = X_train_imputed.fillna(train_medians)
    
    # 5. 최종 학습에 사용할 컬럼 이름 저장 (align_final_features 함수를 위함)
    feature_names = X_train.columns.tolist()
    
    return X_train, Y_train, train_medians, feature_names

# --------------------------------------------------------------------------
# B. 모델 학습 및 저장 함수
# --------------------------------------------------------------------------
def train_and_save_model(X_train, Y_train, model_prefix, params=None):
    """XGBoost 모델을 학습, 보정하고 .pkl 파일로 저장합니다."""
    
    print(f"\n--- {model_prefix} 모델 학습 및 저장 시작 ---")
    
    # 학습/보정 데이터 분할
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, 
                                                Y_train, 
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=Y_train)
    
    # 1. XGBoost 분류기 학습 
    if params is None:
        params = {
            'objective': 'binary:logistic', 
            'n_estimators': 1500, 
            'learning_rate': 0.05,
            'eval_metric': 'auc', 
            'random_state': 42, 
            'n_jobs': -1,
            'early_stopping_rounds' : 50
        }
    else:
        if 'early_stopping_rounds' not in params:
            params['early_stopping_rounds'] = 50

    n_estimators = params.pop('n_estimators', 1500)

    xgb_clf = XGBClassifier(**params)

    if len(X_cal) == 0:
        print("*경고*: 검증 데이터셋이 비어있어 Early Stopping을 건너뛰고 진행합니다.")
        xgb_clf.fit(X_tr, y_tr)
        best_ntree = n_estimators
    else:
        xgb_clf.fit(
            X_tr, y_tr, 
            eval_set=[(X_cal, y_cal)],
            verbose=False
        )

    best_ntree = xgb_clf.best_iteration + 1 if hasattr(xgb_clf, 'best_iteration') else n_estimators 
    print(f"   -> 최적 n_estimators: {best_ntree}회 학습")

    final_params = params.copy()
    final_params.pop('early_stopping_rounds', None)
    final_xgb_clf = XGBClassifier(**final_params, n_estimators=best_ntree)
    final_xgb_clf.fit(X_tr, y_tr) 
    
    # 2. 모델 보정 (Calibration)
    calibrated_model = CalibratedClassifierCV(
        final_xgb_clf, method='isotonic', cv=5, #cv='prefit'
    )
    # calibrated_model.fit(X_cal, y_cal)
    calibrated_model.fit(X_tr, y_tr)
    
    # 3. 모델 저장
    joblib.dump(calibrated_model, os.path.join(MODEL_DIR, f'{model_prefix}_model.pkl'))
    print(f"✅ {model_prefix} 모델이 '{model_prefix}_model.pkl'로 저장되었습니다.")
    
    return calibrated_model

def tune_with_optuna(X_train, Y_train, prefix):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        model = XGBClassifier(**params, early_stopping_rounds=100)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        return auc
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=40, timeout=1200)  # 20분 제한
    
    print(f"\n{prefix} Optuna Best AUC : {study.best_value:.5f}")
    print(f"{prefix} Best params :", study.best_params)
    return study.best_params

# --------------------------------------------------------------------------
# C. 메인 실행 블록
# --------------------------------------------------------------------------
if __name__ == '__main__':
    
    print("1. 데이터 로드 및 전처리 통계량 계산")
    try:
        train_info = pd.read_csv(TRAIN_INFO_PATH)
        train_A = pd.read_csv(TRAIN_A_PATH)
        train_B = pd.read_csv(TRAIN_B_PATH)
    except FileNotFoundError as e:
        print(f"🚨 필수 파일 로드 오류: {e}. 경로를 확인해주세요.")
        exit()

    # Target A 데이터 준비
    X_train_A, Y_train_A, medians_A, features_A = load_and_preprocess_train_data(train_A, train_info, 'A')
    
    # Target B 데이터 준비
    X_train_B, Y_train_B, medians_B, features_B = load_and_preprocess_train_data(train_B, train_info, 'B')
    
    # 통계량 저장 (script.py에서 사용)
    joblib.dump(medians_A, os.path.join(MODEL_DIR, 'train_medians_A.pkl'))
    joblib.dump(medians_B, os.path.join(MODEL_DIR, 'train_medians_B.pkl'))
    print("✅ Train 중간값 통계량이 저장되었습니다.")

    print("\nOptuna 튜닝 시작")
    best_params_A = tune_with_optuna(X_train_A, Y_train_A, "Target_A")
    best_params_B = tune_with_optuna(X_train_B, Y_train_B, "Target_B")

    # 최종 모델 저장
    train_and_save_model(X_train_A, Y_train_A, 'Target_A', params=best_params_A)
    train_and_save_model(X_train_B, Y_train_B, 'Target_B', params=best_params_B)
    