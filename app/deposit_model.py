"""
정기예금 가입 예측 모델 공통 모듈
- 학습(train_model.py), 배치 예측(predict.py), 웹앱(app.py)이 모두 이 파일의 함수를 사용합니다.
- 노트북(project.ipynb)의 전처리·최종 모델과 같은 규칙을 따릅니다.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
NUM_COLS = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
NOM_COLS = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
EDU_COL = ['education']
EDU_ORDER = [['primary', 'secondary', 'tertiary']]
CAL_COLS = ['daily_calls', 'weekly_calls']                 # 캠페인 운영 피처 (6장)
BASE_COLS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
             'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome']
FEATURES = BASE_COLS + CAL_COLS                            # 최종 입력 피처 17개
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


# ── 학습용: 연락 날짜 복원 → 일별·주별 연락 건수 ─────────────────────
def add_campaign_features(raw):
    """날짜순으로 정렬된 원본 데이터에서 연락 날짜를 복원해 daily_calls, weekly_calls를 만든다."""
    df = raw.copy()
    month_num = df['month'].map({m: i + 1 for i, m in enumerate(MONTHS)})
    year = 2008 + (month_num.diff() < -3).cumsum()          # 월이 크게 되돌아가면 다음 해
    date = pd.to_datetime(dict(year=year, month=month_num, day=df['day']))
    df['daily_calls'] = date.groupby(date).transform('size')
    df['weekly_calls'] = date.groupby(date.dt.to_period('W')).transform('size')
    return df


# ── 학습·예측 공통: unknown 처리 규칙 ───────────────────────────────
def fit_cleaning_rules(X_train):
    """Train에서만 계산하는 규칙: 직업별 최빈 학력, 가장 많은 직업"""
    known = X_train[X_train['education'] != 'unknown']
    edu_by_job = known.groupby('job')['education'].agg(lambda s: s.mode()[0]).to_dict()
    return {'edu_by_job': edu_by_job,
            'edu_mode': known['education'].mode()[0],
            'job_mode': X_train.loc[X_train['job'] != 'unknown', 'job'].mode()[0]}


def apply_cleaning_rules(X, rules):
    """
    학습 때와 같은 규칙으로 unknown을 처리한다.
    - 학습 데이터에서는 job unknown 행을 삭제했지만, 예측할 고객은 삭제할 수 없으므로 가장 많은 직업으로 대체
    - education unknown은 직업별 최빈 학력으로 대체
    """
    X = X.copy()
    X['job'] = X['job'].replace('unknown', rules['job_mode'])
    mask = X['education'] == 'unknown'
    X.loc[mask, 'education'] = X.loc[mask, 'job'].map(rules['edu_by_job']).fillna(rules['edu_mode'])
    return X


# ── 최종 모델 파이프라인 ────────────────────────────────────────────
def build_final_pipeline():
    """LightGBM(기본값) + RUS 1:3 + 전처리 (노트북 8장 최종 모델과 동일)"""
    prep = ColumnTransformer([
        ('num', 'passthrough', NUM_COLS + CAL_COLS),
        ('edu', OrdinalEncoder(categories=EDU_ORDER), EDU_COL),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), NOM_COLS)])
    return ImbPipeline([
        ('sampler', RandomUnderSampler(sampling_strategy=1/3, random_state=RANDOM_STATE)),
        ('prep', prep),
        ('model', LGBMClassifier(random_state=RANDOM_STATE, verbose=-1))])


# ── 예측용: 입력 데이터 점검과 준비 ─────────────────────────────────
def prepare_input(customers, rules, daily_calls=None, weekly_calls=None):
    """
    예측할 고객 데이터를 모델 입력 형태로 만든다.
    - daily_calls, weekly_calls 칼럼이 파일에 없으면 캠페인 계획값(인자)으로 채운다.
    - 반환: (모델 입력 DataFrame, 경고 메시지 목록)
    """
    df = customers.copy()
    warnings = []
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'필수 칼럼이 없습니다: {missing}')

    for col, value in [('daily_calls', daily_calls), ('weekly_calls', weekly_calls)]:
        if col not in df.columns:
            if value is None:
                raise ValueError(f'{col} 칼럼이 없으면 캠페인 계획값을 입력해야 합니다.')
            df[col] = value

    if (df['previous'] > 100).any():
        warnings.append(f"previous가 100을 넘는 행 {(df['previous'] > 100).sum()}건: 입력 오류일 수 있습니다.")
    unknown_job = (df['job'] == 'unknown').sum()
    if unknown_job:
        warnings.append(f'job이 unknown인 고객 {unknown_job}명은 가장 많은 직업으로 대체했습니다.')

    df = apply_cleaning_rules(df, rules)
    return df[FEATURES], warnings


def predict_customers(artifact, customers, daily_calls=None, weekly_calls=None):
    """고객 목록에 가입 여부(가입 / 미가입) 예측 결과를 붙여 돌려준다."""
    X, warnings = prepare_input(customers, artifact['rules'], daily_calls, weekly_calls)
    pred = artifact['pipeline'].predict(X)          # 임계값 0.5 기준 0/1
    result = customers.copy()
    result['예측'] = np.where(pred == 1, '가입', '미가입')
    return result, warnings
