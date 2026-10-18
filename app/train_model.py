"""
1단계-1: 최종 모델 학습 및 저장
실행: python train_model.py
결과: model/deposit_model.joblib (모델 + 전처리 규칙 + 성능), sample_customers.csv (앱 시연용)
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from deposit_model import (RANDOM_STATE, BASE_COLS, FEATURES, CAL_COLS, add_campaign_features,
                           fit_cleaning_rules, apply_cleaning_rules, build_final_pipeline)

# 1) 데이터 불러오기 + 연락 건수 피처 (원본 행 순서 그대로 계산)
raw = pd.read_csv('bank-full.csv', sep=';')
raw = add_campaign_features(raw)

# 2) duration 제거 + 중복 제거 (노트북 1장과 같은 기준: duration·연락 건수 칼럼 제외하고 비교)
keep = ~raw.drop(columns=['duration'] + CAL_COLS).duplicated()
df = raw.loc[keep].drop(columns=['duration']).reset_index(drop=True)
X, y = df.drop(columns=['y']), (df['y'] == 'yes').astype(int)

# 3) Train / Validation / Test = 6 : 2 : 2 층화 분할
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RANDOM_STATE)

# 4) Train 정제: job unknown 삭제, 오류값(previous > 100) 삭제, 규칙은 Train에서만 계산
def drop_rows(X_, y_):
    mask = (X_['job'] != 'unknown') & (X_['previous'] <= 100)
    return X_[mask], y_[mask]

X_train, y_train = drop_rows(X_train, y_train)
X_test_clean, y_test_clean = drop_rows(X_test, y_test)
rules = fit_cleaning_rules(X_train)
X_train = apply_cleaning_rules(X_train, rules)
X_test_clean = apply_cleaning_rules(X_test_clean, rules)

# 5) 학습 (Train만 사용, 노트북 8장과 같은 조건)
pipe = build_final_pipeline()
pipe.fit(X_train[FEATURES], y_train)

# 6) Test 성능 확인 (노트북 8-2: F1 0.5114가 나와야 함)
pred = pipe.predict(X_test_clean[FEATURES])
metrics = {'Accuracy': accuracy_score(y_test_clean, pred), 'Precision': precision_score(y_test_clean, pred),
           'Recall': recall_score(y_test_clean, pred), 'F1': f1_score(y_test_clean, pred)}
print(f'Train {len(X_train):,}건으로 학습 / Test {len(X_test_clean):,}건')
print('Test 성능:', {k: round(v, 4) for k, v in metrics.items()})

# 7) 저장: 모델과 전처리 규칙을 한 파일로
os.makedirs('model', exist_ok=True)
joblib.dump({'pipeline': pipe, 'rules': rules, 'features': FEATURES, 'test_metrics': metrics,
             'base_rate': float(y.mean()), 'version': '1.0'}, 'model/deposit_model.joblib')
print('저장 완료: model/deposit_model.joblib')

# 8) 시연용 고객 파일: Test 고객의 기본 정보만 (정답 y와 연락 건수는 제외)
sample = X_test[BASE_COLS].copy()
sample.insert(0, 'customer_id', [f'C{i:05d}' for i in sample.index])
sample.head(500).to_csv('sample_customers.csv', index=False)
print('저장 완료: sample_customers.csv (Test 고객 500명, 시연용)')
