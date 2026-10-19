"""
은행 정기예금 가입 예측 대시보드
- 전처리.ipynb : 데이터 점검, unknown/이상치 실험, 연관성 분석, 파생변수(year, age_group, balance_group)
- 모델_생성.ipynb : 중복 제거, 모델 비교, 불균형 처리, 최종 모델(XGBoost + RUS 1:3), 임계값 결정
- 서비스 기능    : 가입 예측, 캠페인 시뮬레이션, 콜 리스트 우선순위(배치)

실행:  streamlit run app.py
"""

import hashlib
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, f_oneway
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==================================================
# 기본 설정
# ==================================================
st.set_page_config(page_title="정기예금 가입 예측 대시보드", page_icon="🏦", layout="wide")

APP_DIR = Path(__file__).parent
RANDOM_STATE = 42

C_YES = "#1F7A8C"     # 가입(yes)
C_NO = "#BFCCD6"      # 미가입(no)
C_INK = "#13293D"     # 강조
C_WARN = "#C8553D"    # 기준선/선택값

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
RAW_COLS = ["age", "job", "marital", "education", "default", "balance", "housing",
            "loan", "contact", "day", "month", "duration", "campaign", "pdays",
            "previous", "poutcome", "y"]

# 모델_생성.ipynb 피처 정의
NUM_COLS = ["age", "balance", "day", "campaign", "pdays", "previous", "month_contacts"]
CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome", "age_group", "balance_group"]

# 모델_생성.ipynb GridSearchCV(5-fold, PR-AUC) 결과로 선택된 하이퍼파라미터 (중복 제거 후)
BASE_PARAMS = {
    "LogisticRegression": {"C": 0.1},
    "DecisionTree": {"max_depth": 10, "min_samples_leaf": 50},
    "RandomForest": {"max_depth": 10, "min_samples_leaf": 1, "n_estimators": 300},
    "XGBoost": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 300},
    "LightGBM": {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300},
}
# 불균형 처리 후 재튜닝 결과 (평균 순위 상위 3개 조합, 첫 번째가 최종 모델)
TUNED_PARAMS = {
    ("XGBoost", "RUS 1:3"): {"max_depth": 5, "learning_rate": 0.03,
                             "n_estimators": 300, "subsample": 0.8},
    ("LightGBM", "RUS 1:3"): {"num_leaves": 15, "learning_rate": 0.03,
                              "n_estimators": 300, "min_child_samples": 50},
    ("RandomForest", "SMOTENC 1:2"): {"n_estimators": 500, "max_depth": 10,
                                      "min_samples_leaf": 20, "max_features": 0.3},
}
RATIOS = {"1:1": 1.0, "1:2": 1 / 2, "1:3": 1 / 3}
IMB_METHODS = (["처리 전", "class_weight"]
               + [f"RUS {r}" for r in RATIOS] + [f"SMOTENC {r}" for r in RATIOS])


def show(fig, height=None):
    """plotly 그래프 공통 스타일 + 출력"""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Pretendard, 'Noto Sans KR', sans-serif", color=C_INK),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", y=-0.18))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def table(df, fmt="{:.4f}", **kwargs):
    num = df.select_dtypes("number").columns
    st.dataframe(df.style.format({c: fmt for c in num}, na_rep="-"), width="stretch", **kwargs)


# ==================================================
# 1. 데이터 로드 & 파생변수 (전처리.ipynb)
# ==================================================
def read_any_csv(src):
    """세미콜론(bank-full.csv)·콤마(add_bank.csv) 모두 읽기, BOM 제거"""
    raw = src.getvalue() if hasattr(src, "getvalue") else Path(src).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    sep = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [c.strip().strip('"').lstrip("\ufeff") for c in df.columns]
    return df


def balance_thresholds(raw):
    """balance_group 경계값: 0 / 잔고>0 고객의 train 삼분위수 (전처리.ipynb와 같은 분할)"""
    X = raw.drop(columns=["duration", "y"])
    y = (raw["y"] == "yes").astype(int)
    X_tr, _, _, _ = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y)
    q1, q2 = X_tr.loc[X_tr["balance"] > 0, "balance"].quantile([1 / 3, 2 / 3])
    return float(q1), float(q2)


def age_group_of(age):
    return (pd.Series(age).astype(int) // 10 * 10).astype(str) + "대"


def balance_group_of(balance, q1, q2):
    return pd.cut(pd.Series(balance), bins=[-np.inf, 0, q1, q2, np.inf],
                  labels=["le_0", "low", "mid", "high"]).astype(str)


def add_features(raw, q1, q2):
    """전처리.ipynb에서 add_bank.csv를 만든 과정"""
    df = raw.copy()
    df["age_group"] = age_group_of(df["age"]).values
    # 데이터가 시간순 정렬 → 월 숫자가 작아지는 지점이 연도 변경 (2008-05 시작)
    month_num = df["month"].map({m: i + 1 for i, m in enumerate(MONTH_ORDER)})
    df["year"] = 2008 + (month_num < month_num.shift()).cumsum()
    df["balance_group"] = balance_group_of(df["balance"], q1, q2).values
    return df


@st.cache_data(show_spinner="데이터 불러오는 중…")
def load_data(src_key, _src):
    df = read_any_csv(_src)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    raw = df[RAW_COLS].copy()
    q1, q2 = balance_thresholds(raw)
    full = add_features(raw, q1, q2)
    sig = hashlib.md5(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest()
    return raw, full, (q1, q2), sig


def find_default_file():
    for name in ["add_bank.csv", "bank-full.csv"]:
        for base in [APP_DIR, APP_DIR / "data", Path.cwd()]:
            if (base / name).exists():
                return base / name
    return None


# ==================================================
# 2. 모델링 공통 (모델_생성.ipynb)
# ==================================================
def to_model_df(full):
    # duration을 빼면 모든 값이 같은 행 = 같은 고객 중복 입력 → 분할 전에 제거
    # (같은 행이 train과 test에 동시에 들어가는 것 방지)
    d = full[~full.drop(columns="duration").duplicated()].reset_index(drop=True)
    d["y"] = (d["y"] == "yes").astype(int)
    # 같은 연도-월 상담 건수 (캠페인 강도, y 미사용 → 누수 없음)
    d["month_contacts"] = d.groupby(["year", "month"])["month"].transform("size")
    return d


@st.cache_data(show_spinner=False)
def make_splits(sig, _mdf):
    """train 60 / validation 20 / test 20 (stratify)"""
    X, y = _mdf[NUM_COLS + CAT_COLS], _mdf["y"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)
    return dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test)


def make_clf(name, params):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, **params)
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "LightGBM":
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1, **params)
    raise ValueError(name)


def build_model(name, params, method, n_neg, n_pos):
    """처리 전 / class_weight / RUS / SMOTE-NC 조합 생성"""
    clf = make_clf(name, params)
    num_step = StandardScaler() if name == "LogisticRegression" else "passthrough"
    prep = ColumnTransformer([("num", num_step, NUM_COLS),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])

    if method == "class_weight":
        if name == "XGBoost":
            clf.set_params(scale_pos_weight=n_neg / n_pos)
        else:
            clf.set_params(class_weight="balanced")

    steps = [("prep", prep), ("clf", clf)]
    if method in ("처리 전", "class_weight"):
        return Pipeline(steps)

    kind, ratio = method.split()
    if kind == "RUS":
        sampler = RandomUnderSampler(sampling_strategy=RATIOS[ratio], random_state=RANDOM_STATE)
    else:
        sampler = SMOTENC(categorical_features=CAT_COLS, sampling_strategy=RATIOS[ratio],
                          random_state=RANDOM_STATE)
    return ImbPipeline([("sampler", sampler)] + steps)


def metrics_at(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "PR-AUC": average_precision_score(y_true, proba),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


@st.cache_resource(show_spinner=False)
def train_model(sig, name, method, params_items, _sp):
    """(데이터, 모델, 방법, 파라미터)별로 한 번만 학습"""
    y_tr = _sp["y_train"]
    model = build_model(name, dict(params_items), method,
                        int((y_tr == 0).sum()), int((y_tr == 1).sum()))
    start = time.time()
    model.fit(_sp["X_train"], y_tr)
    model.fit_time_ = time.time() - start
    return model


@st.cache_data(show_spinner=False)
def eval_model(sig, name, method, params_items, _sp):
    model = train_model(sig, name, method, params_items, _sp)
    fit_time = getattr(model, "fit_time_", np.nan)
    p_val = model.predict_proba(_sp["X_val"])[:, 1]
    p_tr = model.predict_proba(_sp["X_train"])[:, 1]
    res = metrics_at(_sp["y_val"], p_val, 0.5)
    res["Train PR-AUC"] = average_precision_score(_sp["y_train"], p_tr)
    res["학습 시간(초)"] = fit_time
    return res


# ==================================================
# 3. 통계 함수 (전처리.ipynb)
# ==================================================
def cramers_v(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    t = pd.crosstab(valid["x"], valid["y"])
    if min(t.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.to_numpy().sum()
    phi2 = chi2 / n
    r, k = t.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return np.sqrt(phi2c / denom) if denom > 0 else np.nan


def correlation_ratio(categories, values):
    valid = pd.DataFrame({"c": categories, "v": values}).dropna()
    if valid.empty or valid["c"].nunique() < 2:
        return np.nan
    m = valid["v"].mean()
    g = valid.groupby("c")["v"]
    ss_b = (g.size() * (g.mean() - m) ** 2).sum()
    ss_t = ((valid["v"] - m) ** 2).sum()
    return np.sqrt(ss_b / ss_t) if ss_t else np.nan


@st.cache_data(show_spinner="연관성 계산 중…")
def association_matrices(sig, _raw):
    cats = ["job", "marital", "education", "default", "housing",
            "loan", "contact", "month", "poutcome", "y"]
    nums = ["age", "balance", "day", "campaign", "pdays", "previous"]

    cv = pd.DataFrame(np.eye(len(cats)), index=cats, columns=cats)
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if i < j:
                cv.loc[a, b] = cv.loc[b, a] = cramers_v(_raw[a], _raw[b])

    d = _raw[cats + nums].copy()
    d["pdays"] = d["pdays"].replace(-1, np.nan)     # -1 = 이전 연락 없음 코드값
    eta = pd.DataFrame(index=cats, columns=nums, dtype=float)
    for c in cats:
        for n in nums:
            eta.loc[c, n] = correlation_ratio(d[c], d[n])

    sp = d[nums].corr(method="spearman")
    return cv, eta, sp


def iqr_summary(series, name, exclude=None):
    s = series.dropna()
    if exclude is not None:
        s = s[s != exclude]
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = s[(s < lo) | (s > hi)]
    return {"변수": name, "제외값": str(exclude) if exclude is not None else "-", "대상 수": len(s),
            "Q1": q1, "Q3": q3, "IQR": iqr, "하한": lo, "상한": hi,
            "이상치 수": len(out), "이상치 비율(%)": len(out) / len(s) * 100}


# ==================================================
# 4. 전처리 실험 (전처리.ipynb: unknown 대체 / previous 이상치)
# ==================================================
def prep_baseline_split(raw):
    X = raw.drop(columns=["duration", "y"]).copy()
    y = (raw["y"] == "yes").astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4,
                                                random_state=RANDOM_STATE, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                              random_state=RANDOM_STATE, stratify=y_tmp)
    return X_tr, X_va, y_tr, y_va


def baseline_lr(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), num),
                                                 ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
                     ("model", LogisticRegression(max_iter=1000))])


def score_row(name, y, pred, prob, n_train=None):
    row = {"실험": name}
    if n_train is not None:
        row["Train 크기"] = n_train
    row.update({"PR-AUC": average_precision_score(y, prob),
                "F1": f1_score(y, pred, zero_division=0),
                "Recall": recall_score(y, pred, zero_division=0),
                "Precision": precision_score(y, pred, zero_division=0),
                "Accuracy": accuracy_score(y, pred)})
    return row


@st.cache_data(show_spinner="unknown 대체 실험 실행 중…")
def run_unknown_experiments(sig, _raw):
    X_tr, X_va, y_tr, y_va = prep_baseline_split(_raw)
    base = baseline_lr(X_tr)
    m = clone(base).fit(X_tr, y_tr)
    rows = [score_row("Baseline (unknown 유지)", y_va, m.predict(X_va), m.predict_proba(X_va)[:, 1])]

    def imp_model(nums, cats):
        tr = []
        if nums:
            tr.append(("num", StandardScaler(), nums))
        if cats:
            tr.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
        return Pipeline([("p", ColumnTransformer(tr)), ("m", LogisticRegression(max_iter=1000))])

    jk, ek = X_tr["job"].ne("unknown"), X_tr["education"].ne("unknown")
    job_model = imp_model(["age"], ["education"]).fit(X_tr.loc[jk, ["age", "education"]], X_tr.loc[jk, "job"])
    edu_model = imp_model([], ["job"]).fit(X_tr.loc[ek, ["job"]], X_tr.loc[ek, "education"])

    bins = [-np.inf, 29, 39, 49, 59, np.inf]
    labels = ["20대 이하", "30대", "40대", "50대", "60대 이상"]
    ref = X_tr.copy()
    ref["_ab"] = pd.cut(ref["age"], bins=bins, labels=labels)
    kj, ke = ref[ref["job"].ne("unknown")], ref[ref["education"].ne("unknown")]
    rules = {
        "job_by_edu": kj.groupby("education")["job"].agg(lambda s: s.mode().iloc[0]).to_dict(),
        "job_by_edu_age": kj.groupby(["education", "_ab"], observed=True)["job"]
                            .agg(lambda s: s.mode().iloc[0]).to_dict(),
        "edu_by_job": ke.groupby("job")["education"].agg(lambda s: s.mode().iloc[0]).to_dict(),
        "job_global": kj["job"].mode().iloc[0],
        "edu_global": ke["education"].mode().iloc[0],
    }

    def transform(data, method):
        res = data.copy()
        jm, em = data["job"].eq("unknown"), data["education"].eq("unknown")
        ab = pd.cut(data["age"], bins=bins, labels=labels)

        def job_model_():
            if jm.any():
                res.loc[jm, "job"] = job_model.predict(data.loc[jm, ["age", "education"]])

        def job_mode_():
            res.loc[jm, "job"] = [
                rules["job_by_edu_age"].get((e, a), rules["job_by_edu"].get(e, rules["job_global"]))
                for e, a in zip(data.loc[jm, "education"], ab[jm])]

        def edu_model_():
            if em.any():
                res.loc[em, "education"] = edu_model.predict(data.loc[em, ["job"]])

        def edu_mode_():
            res.loc[em, "education"] = [rules["edu_by_job"].get(j, rules["edu_global"])
                                        for j in data.loc[em, "job"]]

        plan = {"model_both": [job_model_, edu_model_], "mode_both": [job_mode_, edu_mode_],
                "job_model_edu_mode": [job_model_, edu_mode_],
                "job_mode_edu_model": [job_mode_, edu_model_],
                "job_model_only": [job_model_], "job_mode_only": [job_mode_],
                "edu_model_only": [edu_model_], "edu_mode_only": [edu_mode_]}
        for f in plan[method]:
            f()
        return res

    exps = [("Exp1 Model Both", "model_both"), ("Exp2 Mode Both", "mode_both"),
            ("Exp3 Job Model / Edu Mode", "job_model_edu_mode"),
            ("Exp4 Job Mode / Edu Model", "job_mode_edu_model"),
            ("Exp5 Job Model Only", "job_model_only"), ("Exp6 Job Mode Only", "job_mode_only"),
            ("Exp7 Edu Model Only", "edu_model_only"), ("Exp8 Edu Mode Only", "edu_mode_only")]
    for name, method in exps:
        xt, xv = transform(X_tr, method), transform(X_va, method)
        m = clone(base).fit(xt, y_tr)
        rows.append(score_row(name, y_va, m.predict(xv), m.predict_proba(xv)[:, 1]))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="이상치 처리 실험 실행 중…")
def run_outlier_experiments(sig, _raw):
    X_tr, X_va, y_tr, y_va = prep_baseline_split(_raw)
    base = baseline_lr(X_tr)
    nz = X_tr.loc[X_tr["previous"] > 0, "previous"]
    clip_upper = int(np.ceil(nz.quantile(0.99)))
    q1, q3 = nz.quantile([0.25, 0.75])
    iqr_upper = q3 + 1.5 * (q3 - q1)

    xc_tr, xc_va = X_tr.copy(), X_va.copy()
    xc_tr["previous"] = xc_tr["previous"].clip(upper=clip_upper)
    xc_va["previous"] = xc_va["previous"].clip(upper=clip_upper)
    keep_iqr = X_tr["previous"] <= iqr_upper
    keep_max = X_tr["previous"] < X_tr["previous"].max()

    exps = {
        "Baseline (원본)": (X_tr, y_tr, X_va),
        f"P99 Clipping (상한 {clip_upper})": (xc_tr, y_tr, xc_va),
        f"IQR 삭제 (> {iqr_upper:g}, train만)": (X_tr[keep_iqr], y_tr[keep_iqr], X_va),
        "최댓값 행 삭제 (train만)": (X_tr[keep_max], y_tr[keep_max], X_va),
    }
    rows = []
    for name, (xt, yt, xv) in exps.items():
        m = clone(base).fit(xt, yt)
        rows.append(score_row(name, y_va, m.predict(xv), m.predict_proba(xv)[:, 1], len(xt)))
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)


@st.cache_data(show_spinner="피처 구성 실험 실행 중…")
def run_feature_set_experiment(sig, _mdf, _sp):
    """모델_생성.ipynb: year 포함 / 제외 / month_contacts 대체 × 무작위 / 시간순 분할"""
    base_num = [c for c in NUM_COLS if c != "month_contacts"]
    sets = {"year 포함 (기존)": (base_num, CAT_COLS + ["year"]),
            "year 제외": (base_num, CAT_COLS),
            "month_contacts로 대체": (NUM_COLS, CAT_COLS)}
    X_all, y = _mdf[NUM_COLS + CAT_COLS + ["year"]], _mdf["y"]
    n = len(_mdf)
    ct, ce = int(n * 0.6), int(n * 0.8)
    splits = {"무작위 분할": (X_all.loc[_sp["X_train"].index], _sp["y_train"],
                          X_all.loc[_sp["X_val"].index], _sp["y_val"]),
              "시간순 분할": (X_all.iloc[:ct], y.iloc[:ct], X_all.iloc[ct:ce], y.iloc[ct:ce])}
    rows = []
    for split_name, (Xa, ya, Xb, yb) in splits.items():
        for feat_name, (nums, cats) in sets.items():
            for name, params in BASE_PARAMS.items():
                num_step = StandardScaler() if name == "LogisticRegression" else "passthrough"
                pipe = Pipeline([("prep", ColumnTransformer([
                    ("num", num_step, nums), ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])),
                    ("clf", make_clf(name, params))]).fit(Xa, ya)
                proba = pipe.predict_proba(Xb)[:, 1]
                pred = (proba >= 0.5).astype(int)
                rows.append({"모델": name, "분할": split_name, "피처 구성": feat_name,
                             "PR-AUC": average_precision_score(yb, proba),
                             "ROC-AUC": roc_auc_score(yb, proba),
                             "F1": f1_score(yb, pred, zero_division=0),
                             "Recall": recall_score(yb, pred, zero_division=0)})
    return pd.DataFrame(rows)


# ==================================================
# 사이드바: 데이터 소스 & 메뉴
# ==================================================
st.sidebar.title("🏦 정기예금 가입 예측")
st.sidebar.caption("Bank Marketing 데이터 · 전처리 + 모델링 대시보드")

uploaded = st.sidebar.file_uploader("데이터 파일 (bank-full.csv 또는 add_bank.csv)", type="csv")
if uploaded is not None:
    src, src_key = uploaded, f"upload:{uploaded.name}:{uploaded.size}"
else:
    default = find_default_file()
    if default is None:
        st.title("정기예금 가입 예측 대시보드")
        st.info("app.py와 같은 폴더에 `bank-full.csv` 또는 `add_bank.csv`를 두거나, "
                "왼쪽에서 파일을 업로드하면 시작됩니다.")
        st.stop()
    src, src_key = default, f"file:{default}:{default.stat().st_mtime}"

try:
    raw, full, (BAL_Q1, BAL_Q2), SIG = load_data(src_key, src)
except Exception as e:
    st.error(f"데이터를 읽지 못했습니다: {e}")
    st.stop()

mdf = to_model_df(full)
SP = make_splits(SIG, mdf)

PAGES = ["📊 데이터 개요", "🧹 전처리 점검", "🔗 변수 연관성", "🧪 파생변수 근거",
         "🤖 모델 비교", "🎯 최종 모델", "🧮 가입 예측", "📞 캠페인 시뮬레이션", "📋 콜 리스트 우선순위"]
page = st.sidebar.radio("메뉴", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(f"원본 {len(raw):,}행 · 모델링 {len(mdf):,}행 (중복 {len(raw) - len(mdf)}건 제거)\n\n"
                   "기간 2008-05 ~ 2010-11\n\n"
                   f"Train {len(SP['X_train']):,} / Val {len(SP['X_val']):,} / Test {len(SP['X_test']):,}")

BASE_RATE = (raw["y"] == "yes").mean()


# ==================================================
# 페이지 1. 데이터 개요
# ==================================================
def page_overview():
    st.title("데이터 개요")

    with st.expander("필터", expanded=False):
        c1, c2, c3 = st.columns(3)
        years = c1.multiselect("연도", sorted(full["year"].unique()), default=sorted(full["year"].unique()))
        jobs = c2.multiselect("직업", sorted(full["job"].unique()), default=sorted(full["job"].unique()))
        contacts = c3.multiselect("연락 방식", sorted(full["contact"].unique()),
                                  default=sorted(full["contact"].unique()))
    d = full[full["year"].isin(years) & full["job"].isin(jobs) & full["contact"].isin(contacts)]
    if d.empty:
        st.warning("필터 조건에 맞는 고객이 없습니다. 필터를 넓혀 주세요.")
        return
    yb = (d["y"] == "yes").astype(int)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("고객 수", f"{len(d):,}")
    k2.metric("가입자 수", f"{yb.sum():,}")
    k3.metric("가입률", f"{yb.mean():.1%}", f"{(yb.mean() - BASE_RATE) * 100:+.1f}%p vs 전체")
    k4.metric("평균 연락 횟수", f"{d['campaign'].mean():.2f}회")

    # 연도-월 추이
    period = (d.assign(yb=yb).groupby(["year", "month"], sort=False)["yb"]
              .agg(건수="count", 가입률="mean").reset_index())
    period["기간"] = period["year"].astype(str) + "-" + period["month"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=period["기간"], y=period["건수"], name="상담 건수", marker_color=C_NO)
    fig.add_scatter(x=period["기간"], y=period["가입률"], name="가입률", mode="lines+markers",
                    line=dict(color=C_YES, width=2.5), secondary_y=True)
    fig.add_hline(y=BASE_RATE, line_dash="dot", line_color=C_WARN, secondary_y=True,
                  annotation_text=f"전체 평균 {BASE_RATE:.1%}")
    fig.update_yaxes(title_text="상담 건수", secondary_y=False)
    fig.update_yaxes(title_text="가입률", tickformat=".0%", secondary_y=True)
    fig.update_layout(title="연도·월별 상담 건수와 가입률 (시간 순서)")
    show(fig, 420)
    st.caption("2008년은 상담량이 많지만 가입률이 낮고, 2010년은 상담량이 적은 대신 가입률이 높습니다. "
               "이 때문에 모델에서는 year 대신 월별 상담 건수(month_contacts)를 사용합니다.")

    c1, c2 = st.columns(2)
    with c1:
        var = st.selectbox("범주별 가입률", ["age_group", "job", "balance_group", "month", "poutcome",
                                         "contact", "education", "marital", "housing", "loan", "default"])
        g = d.assign(yb=yb).groupby(var)["yb"].agg(가입률="mean", 건수="size").reset_index()
        if var == "month":
            g[var] = pd.Categorical(g[var], MONTH_ORDER, ordered=True)
        elif var == "balance_group":
            g[var] = pd.Categorical(g[var], ["le_0", "low", "mid", "high"], ordered=True)
        g = g.sort_values(var)
        fig = px.bar(g, x=var, y="가입률", text=g["가입률"].map("{:.1%}".format),
                     hover_data={"건수": ":,"}, color_discrete_sequence=[C_YES])
        fig.add_hline(y=BASE_RATE, line_dash="dot", line_color=C_WARN)
        fig.update_layout(yaxis_tickformat=".0%", title=f"{var}별 가입률")
        show(fig, 400)
    with c2:
        num = st.selectbox("수치형 분포", ["age", "balance", "campaign", "pdays", "previous", "day", "duration"])
        clip = st.checkbox("상위 1% 잘라서 보기", value=num in ("balance", "campaign", "previous", "duration"))
        dd = d.copy()
        if clip:
            dd = dd[dd[num] <= dd[num].quantile(0.99)]
        fig = px.histogram(dd, x=num, color="y", barmode="overlay", nbins=50, opacity=0.75,
                           color_discrete_map={"yes": C_YES, "no": C_NO})
        fig.update_layout(title=f"{num} 분포 (가입 여부별)")
        show(fig, 400)
    if num == "duration":
        st.caption("duration은 통화가 끝나야 알 수 있는 값이라 데이터 누수 → 모델 학습에서 제외했습니다.")

    with st.expander("원본 데이터 미리보기"):
        st.dataframe(d.head(200), width="stretch")


# ==================================================
# 페이지 2. 전처리 점검
# ==================================================
def page_preprocess():
    st.title("전처리 점검")
    t1, t2, t3, t4 = st.tabs(["unknown 값", "중복 의심 고객", "이상치", "처리 실험 결과"])

    with t1:
        cnt = raw.eq("unknown").sum()
        uk = pd.DataFrame({"unknown 수": cnt, "비율(%)": cnt / len(raw) * 100})
        uk = uk[uk["unknown 수"] > 0].sort_values("unknown 수", ascending=False)
        c1, c2 = st.columns([1, 1.3])
        with c1:
            table(uk, "{:,.2f}")
        with c2:
            fig = px.bar(uk.reset_index(), x="index", y="비율(%)", text_auto=".1f",
                         color_discrete_sequence=[C_INK])
            fig.update_layout(title="컬럼별 unknown 비율", xaxis_title="")
            show(fig, 320)

        st.subheader("poutcome · previous · pdays 관계")
        checks = pd.DataFrame({
            "검증": ["previous = 0 → poutcome = unknown", "previous = 0 → pdays = -1",
                   "pdays = -1 → previous = 0", "poutcome = unknown → previous = 0"],
            "결과": [(raw.loc[raw.previous == 0, "poutcome"] == "unknown").all(),
                   (raw.loc[raw.previous == 0, "pdays"] == -1).all(),
                   (raw.loc[raw.pdays == -1, "previous"] == 0).all(),
                   (raw.loc[raw.poutcome == "unknown", "previous"] == 0).all()],
        })
        checks["결과"] = checks["결과"].map({True: "✅ 항상 성립", False: "❌ 예외 있음"})
        st.dataframe(checks, width="stretch", hide_index=True)
        st.caption("이전 연락이 없으면 poutcome은 unknown이지만, unknown이라고 해서 이전 연락이 0회인 것은 아닙니다. "
                   "poutcome의 unknown은 '이전 캠페인 없음'이라는 의미가 있으므로 그대로 둡니다.")

        st.subheader("job ↔ education / age 독립성 검정")
        dc = raw[(raw.job != "unknown") & (raw.education != "unknown")]
        chi2, p, dof, exp = chi2_contingency(pd.crosstab(dc.job, dc.education))
        da = raw[raw.job != "unknown"]
        f, pa = f_oneway(*[g["age"].values for _, g in da.groupby("job")])
        c1, c2 = st.columns(2)
        c1.metric("카이제곱 (job × education)", f"{chi2:,.1f}", f"p = {p:.1e}", delta_color="off")
        c2.metric("ANOVA F (직업별 평균 나이)", f"{f:,.1f}", f"p = {pa:.1e}", delta_color="off")
        st.caption("두 검정 모두 p < 0.05 → job은 education·age와 관련 있음. 이를 근거로 unknown 대체 실험을 설계했습니다.")

    with t2:
        cust = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact"]
        sus = raw[raw.duplicated(subset=cust, keep=False)].copy()
        sus["original_row"] = sus.index
        mmap = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
        mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        mstart = {m: sum(mdays[:m - 1]) for m in range(1, 13)}
        sus["doy"] = sus["month"].map(mmap).map(mstart) + sus["day"]
        sus = sus.sort_values(cust + ["original_row"])
        g = sus.groupby(cust, observed=True)
        sus["prev_row"] = g["original_row"].shift(1)
        gap = sus["doy"] - g["doy"].shift(1)
        sus["gap"] = gap.where(gap >= 0, gap + 365)
        valid = sus[sus.prev_row.notna() & (sus.pdays >= 0) & sus.gap.notna()]
        match = (valid.pdays == valid.gap)

        dup_dur = raw.drop(columns="duration").duplicated(keep=False)
        n_drop = int(raw.drop(columns="duration").duplicated().sum())
        st.subheader("① duration만 다른 중복 → 제거")
        d1, d2, d3 = st.columns(3)
        d1.metric("완전 중복 행", f"{raw.duplicated().sum():,}")
        d2.metric("duration 제외 시 중복 (제거)", f"{n_drop:,}건")
        d3.metric("제거 행의 y", " / ".join(f"{k} {v}" for k, v in
                                          raw[raw.drop(columns='duration').duplicated()]["y"]
                                          .value_counts().items()))
        st.dataframe(raw[dup_dur].sort_values(RAW_COLS[:11]).head(40), width="stretch", height=260)
        st.info("통화 시간(duration)만 다르고 나머지가 모두 같은 행은 같은 고객이 중복 입력된 것으로 판단했습니다. "
                "모두 y = no이고 피처가 같은데 라벨이 다른 경우가 없어, 모델링 단계에서 "
                "train/validation/test 분할 전에 제거했습니다.")

        st.subheader("② 고객 특성만 같은 중복 의심 → 유지")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("고객 특성 동일 그룹 수", f"{g.ngroups:,}")
        c2.metric("고객 특성 중복 의심 행", f"{len(sus):,}")
        c3.metric("이전 행이 있는 비교 대상", f"{len(valid):,}")
        c4.metric("pdays = 날짜 차이 일치율", f"{match.mean():.1%}" if len(valid) else "-")
        st.dataframe(sus[["original_row"] + cust + ["day", "month", "campaign", "pdays",
                                                    "previous", "poutcome", "y"]].head(100),
                     width="stretch", hide_index=True)
        st.info("고객 식별자가 없어 같은 고객이라고 확정할 수 없고, 캠페인 이력·결과가 달라 정보가 있으므로 삭제하지 않고 유지했습니다.")

    with t3:
        rows = [iqr_summary(raw["balance"], "balance"), iqr_summary(raw["campaign"], "campaign"),
                iqr_summary(raw["pdays"], "pdays", -1), iqr_summary(raw["previous"], "previous", 0)]
        table(pd.DataFrame(rows), "{:,.2f}", hide_index=True)
        col = st.selectbox("분포 확인", ["balance", "campaign", "pdays", "previous"])
        s = raw[col]
        s = s[s != -1] if col == "pdays" else (s[s != 0] if col == "previous" else s)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(s, nbins=60, color_discrete_sequence=[C_INK])
            fig.update_layout(title=f"{col} 히스토그램", showlegend=False)
            show(fig, 330)
        with c2:
            fig = px.box(x=s, color_discrete_sequence=[C_YES])
            fig.update_layout(title=f"{col} 박스플롯", xaxis_title=col)
            show(fig, 330)
        neg = (raw.balance < 0)
        st.caption(f"음수 잔액 {neg.sum():,}건({neg.mean():.1%}) · 마이너스 통장, 고액 잔고는 실제로 가능한 값이므로 "
                   "오류로 보지 않았습니다. previous는 P99 클리핑·IQR 삭제를 실험했지만 성능 차이가 거의 없었습니다.")

    with t4:
        st.write("Baseline: Logistic Regression (StandardScaler + OneHot), Train 60% / Validation 20% 기준")
        c1, c2 = st.columns(2)
        run_u = c1.button("unknown 대체 실험 실행 (8개)", width="stretch")
        run_o = c2.button("previous 이상치 실험 실행 (4개)", width="stretch")
        if run_u or st.session_state.get("ran_unknown"):
            st.session_state["ran_unknown"] = True
            r = run_unknown_experiments(SIG, raw)
            st.subheader("unknown 대체 방식 비교")
            table(r, hide_index=True)
            fig = px.bar(r, x="실험", y="PR-AUC", text_auto=".4f", color_discrete_sequence=[C_YES])
            fig.update_yaxes(range=[r["PR-AUC"].min() - 0.01, r["PR-AUC"].max() + 0.005])
            show(fig, 360)
            st.caption("대체 방식 간 차이가 매우 작아 unknown은 대체하지 않고 하나의 범주로 유지했습니다.")
        if run_o or st.session_state.get("ran_outlier"):
            st.session_state["ran_outlier"] = True
            st.subheader("previous 이상치 처리 비교")
            table(run_outlier_experiments(SIG, raw), hide_index=True)


# ==================================================
# 페이지 3. 변수 연관성
# ==================================================
def page_association():
    st.title("변수 연관성")
    cv, eta, sp = association_matrices(SIG, raw)
    t1, t2, t3 = st.tabs(["범주 × 범주 (Cramér's V)", "범주 × 수치 (상관비 η)", "수치 × 수치 (Spearman)"])

    with t1:
        m = cv.astype(float).where(np.tril(np.ones(cv.shape, dtype=bool)))
        fig = px.imshow(m, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto")
        fig.update_layout(title="Cramér's V (편향 보정)")
        show(fig, 560)
        ys = cv["y"].drop("y").sort_values(ascending=False)
        st.write("**y와 연관 높은 순**: " + " > ".join(f"{k}({v:.2f})" for k, v in ys.head(5).items()))
        st.caption("month–contact, month–housing이 강하게 엮임 → 캠페인이 특정 달에 특정 연락 방식·고객층에 집중됨.")

    with t2:
        fig = px.imshow(eta.astype(float), text_auto=".2f", color_continuous_scale="Blues",
                        zmin=0, zmax=1, aspect="auto")
        fig.update_layout(title="상관비 η (pdays = -1 제외)", xaxis_title="수치형", yaxis_title="범주형")
        show(fig, 520)
        st.caption("poutcome–previous, job–age, month–pdays, month–day 연관이 큼.")

    with t3:
        m = sp.where(np.tril(np.ones(sp.shape, dtype=bool)))
        fig = px.imshow(m, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(title="Spearman 상관계수")
        show(fig, 480)
        st.caption("수치형 변수끼리의 상관은 대부분 약함 → 다중공선성 우려 낮음.")


# ==================================================
# 페이지 4. 파생변수 근거
# ==================================================
def page_features():
    st.title("파생변수 근거")
    d = raw.copy()
    d["yb"] = (d["y"] == "yes").astype(int)
    base = d["yb"].mean()

    def iv(bins):
        t = pd.crosstab(bins, d["yb"]) + 0.5
        dist = t / t.sum()
        return ((dist[1] - dist[0]) * np.log(dist[1] / dist[0])).sum()

    t1, t2, t3, t4 = st.tabs(["age_group", "balance_group", "campaign 연락 횟수", "year / month_contacts"])

    with t1:
        cands = {
            "연령대 6구간": ([-np.inf, 29, 39, 49, 59, 69, np.inf],
                        ["20대 이하", "30대", "40대", "50대", "60대", "70대 이상"]),
            "연령대 5구간": ([-np.inf, 29, 39, 49, 59, np.inf], ["20대 이하", "30대", "40대", "50대", "60대 이상"]),
            "연령대 4구간": ([-np.inf, 29, 39, 49, np.inf], ["20대 이하", "30대", "40대", "50대 이상"]),
            "10살 단위 (저장된 age_group)": (None, None),
        }
        choice = st.radio("구간 후보", list(cands), horizontal=True, index=1)
        bins, labels = cands[choice]
        s = full["age_group"] if bins is None else pd.cut(d["age"], bins=bins, labels=labels)
        g = d.groupby(s, observed=True)["yb"].agg(가입률="mean", 건수="size").reset_index()
        g.columns = ["구간", "가입률", "건수"]
        if bins is None:
            g = g.sort_values("구간", key=lambda x: x.str.extract(r"(\d+)")[0].astype(int))
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(g, x="구간", y="가입률", text=g["가입률"].map("{:.1%}".format),
                         hover_data={"건수": ":,"}, color_discrete_sequence=[C_YES])
            fig.add_hline(y=base, line_dash="dot", line_color=C_WARN)
            fig.update_layout(yaxis_tickformat=".0%", title=f"{choice} · IV = {iv(s):.3f}")
            show(fig, 380)
        with c2:
            table(g.assign(lift=g["가입률"] / base), "{:,.3f}", hide_index=True)
        st.caption("30세 미만과 60세 이상에서 가입률이 높은 U자형. 6구간은 IV가 가장 높지만 70대 이상 표본이 작아 "
                   "불안정하고, 4구간은 50대와 60대 이상이 섞여 IV가 급감합니다.")

    with t2:
        st.write(f"경계값: **0** (부트스트랩 결정트리에서 항상 등장) / **{BAL_Q1:,.0f}** / **{BAL_Q2:,.0f}** "
                 "(잔고 > 0 고객의 train 삼분위수)")
        g = d.groupby(full["balance_group"])["yb"].agg(가입률="mean", 건수="size")
        g = g.reindex(["le_0", "low", "mid", "high"]).reset_index()
        g.columns = ["구간", "가입률", "건수"]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(g, x="구간", y="가입률", text=g["가입률"].map("{:.1%}".format),
                         color_discrete_sequence=[C_YES])
            fig.add_hline(y=base, line_dash="dot", line_color=C_WARN)
            fig.update_layout(yaxis_tickformat=".0%", title="balance_group별 가입률")
            show(fig, 360)
        with c2:
            ct = pd.crosstab(full["balance_group"], full["year"], values=d["yb"], aggfunc="mean")
            ct = ct.reindex(["le_0", "low", "mid", "high"])
            lift = ct / d.groupby(full["year"])["yb"].mean()
            fig = px.imshow(lift.round(2), text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig.update_layout(title="연도 내 lift (구간 가입률 ÷ 해당 연도 평균)")
            show(fig, 360)
        st.caption("연도 효과를 걷어내도 잔고가 높을수록 가입률이 높아지는 패턴이 유지됩니다.")

    with t3:
        cb = pd.cut(d["campaign"], bins=[0, 1, 2, 3, 4, 5, 10, np.inf],
                    labels=["1회", "2회", "3회", "4회", "5회", "6~10회", "11회 이상"])
        g = d.groupby(cb, observed=True)["yb"].agg(가입률="mean", 건수="size").reset_index()
        fig = px.bar(g, x="campaign", y="가입률", text=g["가입률"].map("{:.1%}".format),
                     hover_data={"건수": ":,"}, color_discrete_sequence=[C_YES])
        fig.add_hline(y=base, line_dash="dot", line_color=C_WARN)
        fig.update_layout(yaxis_tickformat=".0%", title="현재 캠페인 연락 횟수별 가입률", xaxis_title="")
        show(fig, 380)
        rep = d["campaign"] >= 4
        c1, c2, c3 = st.columns(3)
        c1.metric("4회 이상 연락 고객", f"{rep.sum():,}")
        c2.metric("4회 이상 가입률", f"{d.loc[rep, 'yb'].mean():.1%}")
        c3.metric("전체 평균 대비", f"{d.loc[rep, 'yb'].mean() / base:.2f}배")
        st.caption("연락을 많이 할수록 가입률이 떨어짐 → 3회 연락 후 반응이 없는 고객은 우선순위를 낮추는 정책 제안.")

    with t4:
        y_rate = d.groupby(full["year"])["yb"].agg(가입률="mean", 건수="size")
        table(y_rate, "{:,.3f}")
        mc = mdf.groupby(["year", "month"], sort=False).agg(month_contacts=("month_contacts", "first"),
                                                             가입률=("y", "mean")).reset_index()
        fig = px.scatter(mc, x="month_contacts", y="가입률", color=mc["year"].astype(str),
                         hover_data=["month"], size_max=12,
                         color_discrete_sequence=[C_NO, C_INK, C_YES])
        fig.update_traces(marker=dict(size=11))
        fig.update_layout(yaxis_tickformat=".0%", title="월별 상담 건수 vs 가입률", legend_title="연도")
        show(fig, 400)
        st.caption("year는 미래 데이터에서 쓸 수 없는 '시점'을 외우는 변수라 제외하고, "
                   "같은 연도-월의 상담 건수(month_contacts)로 캠페인 강도를 대신 반영했습니다.")

        st.subheader("피처 구성 × 분할 방식 실험")
        st.caption("튜닝된 5개 모델로 year 포함 / year 제외 / month_contacts로 대체를 비교합니다. "
                   "시간순 분할 = 앞 60% 학습, 다음 20% 평가 (임계값 0.5).")
        if st.button("피처 구성 실험 실행 (30회 학습)") or st.session_state.get("ran_feat"):
            st.session_state["ran_feat"] = True
            fr = run_feature_set_experiment(SIG, mdf, SP)
            metric = st.radio("지표", ["PR-AUC", "ROC-AUC", "F1", "Recall"], horizontal=True, key="feat_m")
            fig = px.bar(fr, x="모델", y=metric, color="피처 구성", barmode="group", facet_col="분할",
                         text_auto=".3f", color_discrete_sequence=[C_NO, C_INK, C_YES])
            fig.update_layout(title=f"{metric} · 피처 구성별 비교")
            show(fig, 440)
            table(fr.pivot_table(index="모델", columns=["분할", "피처 구성"], values=metric, sort=False))
            st.caption("무작위 분할에서는 month_contacts로 대체해도 year 포함과 비슷하거나 더 좋습니다. "
                       "시간순 분할(2008년으로 학습 → 2009년 평가)에서는 모든 구성의 성능이 크게 떨어져, "
                       "기간이 바뀌면 일반화가 어렵다는 한계도 함께 확인됩니다.")


# ==================================================
# 페이지 5. 모델 비교
# ==================================================
def page_models():
    st.title("모델 비교")
    st.caption("하이퍼파라미터는 모델_생성.ipynb의 5-fold GridSearchCV(PR-AUC 기준) 결과를 고정값으로 사용합니다. "
               "평가 기준은 Validation, 임계값 0.5.")
    t1, t2 = st.tabs(["튜닝된 Baseline 5종", "불균형 처리 조합"])

    with t1:
        with st.expander("모델별 하이퍼파라미터"):
            st.json(BASE_PARAMS)
        if st.button("5개 모델 학습·평가", type="primary") or st.session_state.get("ran_base"):
            st.session_state["ran_base"] = True
            rows, bar = [], st.progress(0.0)
            for i, (name, params) in enumerate(BASE_PARAMS.items()):
                bar.progress(i / len(BASE_PARAMS), text=f"{name} 학습 중…")
                r = eval_model(SIG, name, "처리 전", tuple(sorted(params.items())), SP)
                rows.append({"모델": name, **r})
            bar.empty()
            res = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
            table(res[["모델", "PR-AUC", "ROC-AUC", "F1", "Recall", "Precision", "Accuracy",
                       "Train PR-AUC", "TP", "FP", "학습 시간(초)"]], hide_index=True)
            long = res.melt(id_vars="모델", value_vars=["PR-AUC", "F1", "Recall", "Precision"],
                            var_name="지표", value_name="점수")
            fig = px.bar(long, x="지표", y="점수", color="모델", barmode="group", text_auto=".3f",
                         color_discrete_sequence=[C_YES, C_INK, "#6C8EAD", "#9DB4C0", "#E0A458"])
            fig.update_layout(title="모델별 성능 (Validation)")
            show(fig, 420)
            st.info("정확도는 89~90%로 비슷하지만 Recall이 18~30% → 클래스 불균형(yes 11.7%) 때문에 "
                    "가입 고객 대부분을 놓칩니다. 다음 탭에서 불균형 처리를 비교합니다.")

    with t2:
        c1, c2 = st.columns(2)
        models = c1.multiselect("모델", list(BASE_PARAMS), default=["LightGBM", "XGBoost", "RandomForest"])
        methods = c2.multiselect("불균형 처리 방법", IMB_METHODS,
                                 default=["처리 전", "class_weight", "RUS 1:1", "RUS 1:2", "RUS 1:3"])
        st.caption("SMOTE-NC는 학습 시간이 길 수 있습니다. 한 번 학습한 조합은 캐시되어 바로 표시됩니다.")
        if st.button("선택 조합 실행", type="primary") and models and methods:
            rows, total = [], len(models) * len(methods)
            bar = st.progress(0.0)
            for i, (meth, name) in enumerate([(m, n) for m in methods for n in models]):
                bar.progress(i / total, text=f"{name} | {meth} ({i + 1}/{total})")
                r = eval_model(SIG, name, meth, tuple(sorted(BASE_PARAMS[name].items())), SP)
                rows.append({"모델": name, "방법": meth, **r})
            bar.empty()
            st.session_state["imb_res"] = pd.DataFrame(rows)

        res = st.session_state.get("imb_res")
        if res is not None:
            metric = st.radio("지표", ["PR-AUC", "Recall", "Precision", "F1", "학습 시간(초)"], horizontal=True)
            pv = res.pivot(index="모델", columns="방법", values=metric)
            pv = pv[[m for m in IMB_METHODS if m in pv.columns]]
            fig = px.imshow(pv, text_auto=".3f", aspect="auto",
                            color_continuous_scale="Blues" if metric != "학습 시간(초)" else "Greys")
            fig.update_layout(title=f"{metric} · 모델 × 불균형 처리 방법")
            show(fig, 120 + 60 * len(pv))

            rank_cols = ["Accuracy", "Precision", "Recall", "F1"]
            res = res.assign(평균순위=res[rank_cols].rank(ascending=False).mean(axis=1))
            st.subheader("평균 순위 상위 조합")
            table(res.sort_values(["평균순위", "PR-AUC"], ascending=[True, False])
                  [["모델", "방법", "평균순위", "PR-AUC", "F1", "Recall", "Precision", "Accuracy", "TP", "FP"]]
                  .head(10), hide_index=True)
            st.caption("노트북 결과: 모델별 최상위 조합은 LightGBM | RUS 1:3, XGBoost | RUS 1:3, "
                       "RandomForest | SMOTE 1:2 → 이 세 조합을 재튜닝해 최종 모델을 골랐습니다.")


# ==================================================
# 페이지 6. 최종 모델
# ==================================================
def get_final(key):
    name, method = key
    params = TUNED_PARAMS[key]
    with st.spinner(f"{name} | {method} 학습 중…"):
        model = train_model(SIG, name, method, tuple(sorted(params.items())), SP)
    return model


@st.cache_data(show_spinner=False)
def final_probas(sig, key, _model):
    return (_model.predict_proba(SP["X_val"])[:, 1], _model.predict_proba(SP["X_test"])[:, 1])


def best_threshold(y, proba, rule, min_recall):
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]
    f1s = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
    if rule == "F1 최대":
        i = int(np.argmax(f1s))
    else:
        ok = np.where(rec >= min_recall)[0]
        i = int(ok[np.argmax(prec[ok])]) if len(ok) else int(np.argmax(f1s))
    return float(thr[i]), prec, rec, thr, f1s


def page_final():
    st.title("최종 모델")
    keys = list(TUNED_PARAMS)
    key = st.selectbox("모델 조합 (불균형 처리 후 재튜닝)", keys,
                       format_func=lambda k: f"{k[0]} | {k[1]}", index=0)
    st.session_state["final_key"] = key
    model = get_final(key)

    with st.expander("최종 모델 선정 과정 (재튜닝 전/후 비교)"):
        st.caption("평균 순위 상위 3개 조합을 5-fold GridSearchCV로 재튜닝 → 조합마다 튜닝 전/후 중 "
                   "Validation PR-AUC가 높은 쪽 선택 → 그중 PR-AUC가 가장 높은 조합이 최종 모델 (임계값 0.5 기준)")
        if st.button("튜닝 전/후 비교 실행") or st.session_state.get("ran_sel"):
            st.session_state["ran_sel"] = True
            rows = []
            with st.spinner("학습 중… (RandomForest + SMOTE-NC는 시간이 조금 걸립니다)"):
                for (n_, m_), tp in TUNED_PARAMS.items():
                    b = eval_model(SIG, n_, m_, tuple(sorted(BASE_PARAMS[n_].items())), SP)
                    a = eval_model(SIG, n_, m_, tuple(sorted(tp.items())), SP)
                    for stage, r in [("튜닝 전", b), ("튜닝 후", a)]:
                        rows.append({"조합": f"{n_} | {m_}", "구분": stage, **{k: r[k] for k in
                                     ["PR-AUC", "Recall", "Precision", "F1", "TP", "FP"]}})
            sel = pd.DataFrame(rows)
            table(sel, hide_index=True)
            fig = px.bar(sel, x="조합", y="PR-AUC", color="구분", barmode="group", text_auto=".4f",
                         color_discrete_map={"튜닝 전": C_NO, "튜닝 후": C_YES})
            fig.update_yaxes(range=[sel["PR-AUC"].min() - 0.01, sel["PR-AUC"].max() + 0.005])
            show(fig, 340)

    p_val, p_test = final_probas(SIG, key, model)
    y_val, y_test = SP["y_val"], SP["y_test"]

    c1, c2, c3 = st.columns([1, 1, 2])
    rule = c1.radio("임계값 기준", ["F1 최대", "Recall 목표"])
    min_recall = c2.slider("최소 Recall", 0.3, 0.95, 0.70, 0.05, disabled=rule != "Recall 목표")
    auto_thr, prec, rec, thr, f1s = best_threshold(y_val, p_val, rule, min_recall)
    thr_sel = c3.slider("임계값 (yes 확률이 이 값 이상이면 가입 예측)", 0.05, 0.95,
                        float(round(auto_thr, 3)), 0.005, key=f"thr_{key}_{rule}_{min_recall}")
    st.session_state["threshold"] = thr_sel

    mv, mt = metrics_at(y_val, p_val, thr_sel), metrics_at(y_test, p_test, thr_sel)
    k = st.columns(5)
    for col, m in zip(k, ["PR-AUC", "ROC-AUC", "Recall", "Precision", "F1"]):
        col.metric(f"Test {m}", f"{mt[m]:.3f}", f"{mt[m] - mv[m]:+.3f} vs Val", delta_color="off")

    t1, t2, t3, t4 = st.tabs(["임계값 분석", "혼동행렬", "타깃 마케팅 효율", "변수 중요도"])

    with t1:
        fig = go.Figure()
        fig.add_scatter(x=thr, y=prec, name="Precision", line=dict(color=C_INK))
        fig.add_scatter(x=thr, y=rec, name="Recall", line=dict(color=C_YES))
        fig.add_scatter(x=thr, y=f1s, name="F1", line=dict(color="#E0A458", width=3))
        fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="0.5")
        fig.add_vline(x=thr_sel, line_dash="dash", line_color=C_WARN,
                      annotation_text=f"선택 {thr_sel:.3f}")
        fig.update_layout(title="임계값별 Precision / Recall / F1 (Validation)",
                          xaxis_title="임계값", yaxis_title="점수")
        show(fig, 420)
        cmp = pd.DataFrame({"Val · 0.5": metrics_at(y_val, p_val, 0.5),
                            f"Val · {thr_sel:.3f}": mv, f"Test · {thr_sel:.3f}": mt}).T
        table(cmp, "{:,.4f}")

    with t2:
        c1, c2 = st.columns(2)
        for col, (title, yt, p) in zip([c1, c2], [("Validation", y_val, p_val), ("Test", y_test, p_test)]):
            cm = confusion_matrix(yt, (p >= thr_sel).astype(int))
            fig = px.imshow(cm, text_auto=",", color_continuous_scale="Blues",
                            x=["예측 no", "예측 yes"], y=["실제 no", "실제 yes"])
            fig.update_layout(title=f"{title} 혼동행렬 (실제 yes {int(yt.sum()):,}명)",
                              coloraxis_showscale=False)
            with col:
                show(fig, 380)
        st.caption("실제 yes·예측 no 칸(FN)이 놓친 가입 고객, 실제 no·예측 yes 칸(FP)이 헛걸음한 연락입니다.")

    with t3:
        order = np.argsort(-p_test)
        yt = y_test.values[order]
        pct = np.arange(1, len(yt) + 1) / len(yt)
        captured = np.cumsum(yt) / yt.sum()
        fig = go.Figure()
        fig.add_scatter(x=pct, y=captured, name="모델 순서대로 연락", line=dict(color=C_YES, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="무작위 연락", line=dict(color="gray", dash="dot"))
        fig.update_layout(title="누적 가입자 포착률 (Test)", xaxis_title="연락한 고객 비율",
                          yaxis_title="찾아낸 가입자 비율", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        show(fig, 420)
        dec = pd.DataFrame({"p": p_test, "y": y_test.values})
        dec["분위"] = pd.qcut(dec["p"].rank(method="first", ascending=False), 10,
                            labels=[f"상위 {i * 10}%" for i in range(1, 11)])
        lt = dec.groupby("분위", observed=True).agg(고객수=("y", "size"), 가입자=("y", "sum"),
                                                     가입률=("y", "mean"), 평균확률=("p", "mean"))
        lt["lift"] = lt["가입률"] / dec["y"].mean()
        lt["누적 포착률"] = lt["가입자"].cumsum() / lt["가입자"].sum()
        table(lt, "{:,.3f}")
        top = st.slider("상위 몇 %에게 연락할까요?", 5, 100, 20, 5)
        n = int(len(yt) * top / 100)
        st.success(f"확률 상위 {top}% ({n:,}명)에게만 연락하면 Test 가입자 {int(yt.sum()):,}명 중 "
                   f"**{int(yt[:n].sum()):,}명({yt[:n].sum() / yt.sum():.1%})**을 찾을 수 있습니다. "
                   f"(무작위라면 약 {top}%)")

    with t4:
        st.caption("Permutation importance: Validation에서 변수 값을 섞었을 때 PR-AUC가 얼마나 떨어지는지")
        if st.button("변수 중요도 계산"):
            with st.spinner("계산 중… (수십 초 걸릴 수 있습니다)"):
                pi = permutation_importance(model, SP["X_val"], y_val, scoring="average_precision",
                                            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
            st.session_state[f"pi_{key}"] = pd.DataFrame(
                {"변수": SP["X_val"].columns, "중요도": pi.importances_mean, "표준편차": pi.importances_std}
            ).sort_values("중요도")
        imp = st.session_state.get(f"pi_{key}")
        if imp is not None:
            fig = px.bar(imp, x="중요도", y="변수", orientation="h", error_x="표준편차",
                         color_discrete_sequence=[C_YES])
            fig.update_layout(title="PR-AUC 감소량")
            show(fig, 520)


# ==================================================
# 페이지 7. 가입 예측
# ==================================================
def build_features(df_in, mc_override=None):
    """원본 형식 고객 데이터 → 모델 입력 피처 (mc_override: month_contacts를 한 값으로 지정)"""
    d = df_in.copy()
    d["age_group"] = age_group_of(d["age"]).values
    d["balance_group"] = balance_group_of(d["balance"], BAL_Q1, BAL_Q2).values
    lookup = mdf.groupby(["year", "month"])["month_contacts"].first()
    by_month = mdf.groupby("month")["month_contacts"].mean()
    if "year" in d.columns:
        d["month_contacts"] = [lookup.get((y, m), by_month.get(m, mdf["month_contacts"].median()))
                               for y, m in zip(d["year"], d["month"])]
    else:
        d["month_contacts"] = d["month"].map(by_month).fillna(mdf["month_contacts"].median())
    if mc_override is not None:
        d["month_contacts"] = int(mc_override)
    return d[NUM_COLS + CAT_COLS]


def page_predict():
    st.title("가입 예측")
    key = st.session_state.get("final_key", list(TUNED_PARAMS)[0])
    model = get_final(key)
    if "threshold" not in st.session_state:
        p_val, _ = final_probas(SIG, key, model)
        st.session_state["threshold"] = best_threshold(SP["y_val"], p_val, "F1 최대", 0.7)[0]
    thr = st.session_state["threshold"]
    st.caption(f"모델: {key[0]} | {key[1]} · 임계값 {thr:.3f} (최종 모델 메뉴에서 변경)")

    st.caption("여러 고객을 한 번에 점수화하려면 📋 콜 리스트 우선순위 메뉴를 이용하세요.")
    t1 = st.container()
    opts = {c: sorted(raw[c].unique()) for c in ["job", "marital", "education", "default",
                                                 "housing", "loan", "contact", "poutcome"]}
    with t1:
        with st.form("one"):
            c1, c2, c3, c4 = st.columns(4)
            age = c1.number_input("나이", 18, 95, 35)
            job = c2.selectbox("직업", opts["job"], index=opts["job"].index("management"))
            marital = c3.selectbox("결혼", opts["marital"], index=opts["marital"].index("married"))
            education = c4.selectbox("학력", opts["education"], index=opts["education"].index("tertiary"))
            balance = c1.number_input("잔고", -10000, 120000, 1500, step=100)
            default = c2.selectbox("신용불량", opts["default"])
            housing = c3.selectbox("주택대출", opts["housing"])
            loan = c4.selectbox("개인대출", opts["loan"])
            contact = c1.selectbox("연락 방식", opts["contact"])
            year = c2.selectbox("연도", sorted(full["year"].unique()), index=0)
            month = c3.selectbox("월", MONTH_ORDER, index=4)
            day = c4.number_input("일", 1, 31, 15)
            campaign = c1.number_input("이번 캠페인 연락 횟수", 1, 70, 1)
            previous = c2.number_input("이전 캠페인 연락 횟수", 0, 300, 0)
            pdays = c3.number_input("이전 연락 후 경과일 (-1 = 없음)", -1, 900, -1)
            poutcome = c4.selectbox("이전 캠페인 결과", opts["poutcome"],
                                    index=opts["poutcome"].index("unknown"))
            ok = st.form_submit_button("가입 확률 예측", type="primary")
        if ok:
            row = pd.DataFrame([dict(age=age, job=job, marital=marital, education=education, default=default,
                                     balance=balance, housing=housing, loan=loan, contact=contact, day=day,
                                     month=month, campaign=campaign, pdays=pdays, previous=previous,
                                     poutcome=poutcome, year=year)])
            p = float(model.predict_proba(build_features(row))[:, 1][0])
            c1, c2 = st.columns([1, 1.4])
            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=p * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": C_YES if p >= thr else C_INK},
                           "threshold": {"line": {"color": C_WARN, "width": 4}, "value": thr * 100}}))
                fig.update_layout(title="가입 확률")
                show(fig, 300)
            with c2:
                if p >= thr:
                    st.success(f"**연락 추천** · 확률 {p:.1%} ≥ 임계값 {thr:.1%}")
                else:
                    st.warning(f"**우선순위 낮음** · 확률 {p:.1%} < 임계값 {thr:.1%}")
                st.write(f"전체 평균 가입률은 {BASE_RATE:.1%}이므로 이 고객은 평균의 **{p / BASE_RATE:.1f}배** 수준입니다.")
                st.caption("RUS로 학습한 모델은 확률이 실제 가입률보다 높게 나오는 경향이 있어, "
                           "절대값보다 고객 간 순위로 해석하는 것이 안전합니다.")




# ==================================================
# 페이지 8. 캠페인 시뮬레이션 (month_contacts = 은행이 정하는 값)
# ==================================================
MC_MIN, MC_MAX = 10, 8000
SIM_N = 500


@st.cache_data(show_spinner=False)
def sim_sample(sig):
    """테스트셋에서 500건 층화 추출 (가입 비율 유지, 항상 같은 고객)"""
    X_s, _, y_s, _ = train_test_split(SP["X_test"], SP["y_test"], train_size=SIM_N,
                                      stratify=SP["y_test"], random_state=RANDOM_STATE)
    return X_s, y_s


@st.cache_data(show_spinner=False)
def sim_curve(sig, key, thr, _model, _X):
    """month_contacts를 바꿔 가며 가입 예측 인원 계산 (민감도 곡선)"""
    grid = np.unique(np.concatenate([np.linspace(MC_MIN, MC_MAX, 81),
                                     mdf["month_contacts"].unique()])).astype(int)
    grid = grid[(grid >= MC_MIN) & (grid <= MC_MAX)]
    rows = []
    for v in grid:
        p = _model.predict_proba(_X.assign(month_contacts=v))[:, 1]
        rows.append({"month_contacts": v, "가입 예측": int((p >= thr).sum()), "평균 확률": p.mean()})
    return pd.DataFrame(rows)


def page_campaign():
    st.title("캠페인 시뮬레이션")
    st.write("고객 정보는 그대로 두고, 은행이 정하는 **월별 상담 건수(month_contacts)**만 바꿨을 때 "
             f"테스트 고객 {SIM_N}명의 가입 예측이 어떻게 달라지는지 봅니다.")

    key = st.session_state.get("final_key", list(TUNED_PARAMS)[0])
    model = get_final(key)
    if "threshold" not in st.session_state:
        p_val, _ = final_probas(SIG, key, model)
        st.session_state["threshold"] = best_threshold(SP["y_val"], p_val, "F1 최대", 0.7)[0]
    thr = st.session_state["threshold"]

    X_s, y_s = sim_sample(SIG)
    p_orig = model.predict_proba(X_s)[:, 1]

    c1, c2 = st.columns([1, 2])
    mc = c1.number_input("이번 달 상담 건수 (month_contacts)", MC_MIN, MC_MAX, 1000, step=10)
    c2.caption(f"모델: {key[0]} | {key[1]} · 임계값 {thr:.3f}\n\n"
               f"학습 데이터의 실제 범위는 {mdf['month_contacts'].min():,}~{mdf['month_contacts'].max():,}건입니다. "
               "트리 모델이라 이 범위 밖의 값은 양 끝 값과 같은 결과가 나옵니다.")

    p_new = model.predict_proba(X_s.assign(month_contacts=mc))[:, 1]
    yes_new, yes_orig = p_new >= thr, p_orig >= thr

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("가입 예측", f"{yes_new.sum():,}명", f"{int(yes_new.sum() - yes_orig.sum()):+,} vs 원래 값")
    k2.metric("미가입 예측", f"{(~yes_new).sum():,}명")
    k3.metric("평균 가입 확률", f"{p_new.mean():.1%}", f"{(p_new.mean() - p_orig.mean()) * 100:+.1f}%p")
    k4.metric("예측이 바뀐 고객", f"{(yes_new != yes_orig).sum():,}명",
              f"미가입→가입 {(yes_new & ~yes_orig).sum()} · 가입→미가입 {(~yes_new & yes_orig).sum()}",
              delta_color="off")

    st.warning("과거 데이터에서 상담량이 적었던 시기(주로 2009~2010년)는 가입률 자체가 높았습니다. "
               "모델은 이 패턴을 반영할 뿐이므로, 상담을 줄이면 실제로 가입이 늘어난다는 인과 관계로 해석하면 안 됩니다.")

    # 민감도 곡선
    curve = sim_curve(SIG, key, round(thr, 6), model, X_s)
    fig = go.Figure()
    fig.add_scatter(x=curve["month_contacts"], y=curve["가입 예측"], mode="lines",
                    line=dict(color=C_YES, width=3, shape="hv"), name="가입 예측 인원")
    fig.add_vline(x=mc, line_dash="dash", line_color=C_WARN, annotation_text=f"입력값 {mc:,}")
    fig.add_hline(y=int(yes_orig.sum()), line_dash="dot", line_color="gray",
                  annotation_text=f"원래 값 기준 {int(yes_orig.sum())}명")
    fig.update_layout(title=f"상담 건수별 가입 예측 인원 (테스트 {SIM_N}명)",
                      xaxis_title="month_contacts", yaxis_title="가입 예측 인원")
    show(fig, 380)

    # 고객 목록
    view = X_s[["age", "job", "marital", "education", "balance", "housing", "loan", "contact",
                "month", "campaign", "previous", "poutcome", "month_contacts"]].rename(
        columns={"month_contacts": "원래 상담 건수"}).copy()
    view["가입 확률"] = p_new
    view["원래 확률"] = p_orig
    view["예측 변화"] = np.select([yes_new & ~yes_orig, ~yes_new & yes_orig], ["미가입→가입", "가입→미가입"], "-")
    view["실제 y (참고)"] = np.where(y_s.values == 1, "yes", "no")
    view = view.sort_values("가입 확률", ascending=False)

    fmt = {"가입 확률": "{:.1%}", "원래 확률": "{:.1%}", "balance": "{:,}", "원래 상담 건수": "{:,}"}
    t1, t2 = st.tabs([f"✅ 가입 예측 ({yes_new.sum()}명)", f"❌ 미가입 예측 ({(~yes_new).sum()}명)"])
    with t1:
        st.dataframe(view[view["가입 확률"] >= thr].style.format(fmt), width="stretch", height=420)
    with t2:
        st.dataframe(view[view["가입 확률"] < thr].style.format(fmt), width="stretch", height=420)
    st.caption("실제 y는 원래 상담 건수일 때의 결과라, 상담 건수를 바꾼 상황의 정답이 아닙니다. 참고용으로만 보세요.")

    st.download_button("시뮬레이션 결과 CSV 다운로드", view.to_csv(index=False).encode("utf-8-sig"),
                       f"campaign_sim_{mc}.csv", "text/csv")


# ==================================================
# 페이지 9. 콜 리스트 우선순위 (배치) — 캠페인 기획자 · 상담팀장
# ==================================================
INPUT_COLS = [c for c in RAW_COLS if c not in ("duration", "y")]
ID_CANDIDATES = ["고객번호", "고객ID", "customer_id", "client_id", "id", "ID"]
NUMERIC_INPUTS = ["age", "balance", "day", "campaign", "pdays", "previous"]


def current_model():
    """최종 모델 메뉴에서 고른 모델·임계값 (없으면 기본값)"""
    key = st.session_state.get("final_key", list(TUNED_PARAMS)[0])
    model = get_final(key)
    if "threshold" not in st.session_state:
        p_val, _ = final_probas(SIG, key, model)
        st.session_state["threshold"] = best_threshold(SP["y_val"], p_val, "F1 최대", 0.7)[0]
    return key, model, st.session_state["threshold"]


@st.cache_resource(show_spinner=False)
def calibrator(sig, key, _model):
    """모델 점수 → 실제 가입률 보정 (Validation 기준 Platt scaling)
    RUS·SMOTE로 학습한 모델은 점수가 실제 가입률보다 높게 나오므로, 예상 가입자 수는 보정값으로 계산"""
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)
    p_val = _model.predict_proba(SP["X_val"])[:, 1]
    lr = LogisticRegression().fit(logit(p_val), SP["y_val"])
    return lambda p: lr.predict_proba(logit(np.asarray(p)))[:, 1]


@st.cache_data(show_spinner=False)
def demo_list(sig):
    """체험용: 테스트셋 고객 (모델이 학습에 쓰지 않은 고객)"""
    d = mdf.loc[SP["X_test"].index, INPUT_COLS].reset_index(drop=True)
    d.insert(0, "고객번호", [f"T{i:05d}" for i in range(1, len(d) + 1)])
    y = mdf.loc[SP["X_test"].index, "y"].reset_index(drop=True)
    return d, y


def template_csv():
    d = demo_list(SIG)[0].head(20)
    return d.to_csv(index=False).encode("utf-8-sig")


def page_calllist():
    st.title("콜 리스트 우선순위")
    st.write("고객 명단을 올리면 가입 점수 순으로 정렬하고, **오늘 통화할 수 있는 건수만큼** 우선 연락 대상을 뽑습니다.")

    key, model, thr = current_model()
    cal = calibrator(SIG, key, model)
    st.caption(f"모델: {key[0]} | {key[1]} · 임계값 {thr:.3f} (최종 모델 메뉴에서 변경)")

    # ---------- 1. 명단 불러오기 ----------
    c1, c2 = st.columns([2, 1])
    with c1:
        f = st.file_uploader("고객 명단 CSV", type="csv", key="calllist_file",
                             help="필수 컬럼: " + ", ".join(INPUT_COLS) + " (고객번호 컬럼은 있으면 그대로 사용)")
    with c2:
        st.download_button("📄 업로드 양식 받기 (예시 20건)", template_csv(),
                           "call_list_template.csv", "text/csv", width="stretch")
        use_demo = st.toggle("테스트셋 고객으로 체험", value=f is None,
                             help="모델이 학습에 사용하지 않은 테스트 고객 9천여 명으로 기능을 체험합니다.")

    y_true = None
    if f is not None and not use_demo:
        try:
            bd = read_any_csv(f)
        except Exception as e:
            st.error(f"파일을 읽지 못했습니다: {e}")
            return
        source = f"업로드: {f.name}"
    elif use_demo:
        bd, y_true = demo_list(SIG)
        bd = bd.copy()
        source = "테스트셋 고객 (체험)"
    else:
        st.info("고객 명단 CSV를 올리거나 '테스트셋 고객으로 체험'을 켜 주세요.")
        return

    miss = [c for c in INPUT_COLS if c not in bd.columns]
    if miss:
        st.error(f"필수 컬럼이 없습니다: {', '.join(miss)} · 업로드 양식을 참고해 주세요.")
        return

    id_col = next((c for c in ID_CANDIDATES if c in bd.columns), None)
    if id_col is None:
        id_col = "고객번호"
        bd.insert(0, id_col, [f"R{i:05d}" for i in range(1, len(bd) + 1)])

    for c in NUMERIC_INPUTS:
        bd[c] = pd.to_numeric(bd[c], errors="coerce")
    for c in [c for c in INPUT_COLS if c not in NUMERIC_INPUTS]:
        bd[c] = bd[c].astype(str).str.strip().str.lower()
    bad = bd[INPUT_COLS].isna().any(axis=1)
    if bad.any():
        st.warning(f"값이 비어 있거나 숫자가 아닌 {bad.sum():,}건은 제외했습니다.")
        bd = bd[~bad].reset_index(drop=True)
        if y_true is not None:
            y_true = y_true[~bad.values].reset_index(drop=True)
    if bd.empty:
        st.error("점수를 계산할 수 있는 고객이 없습니다.")
        return
    bd[NUMERIC_INPUTS] = bd[NUMERIC_INPUTS].astype(int)

    # ---------- 2. 추출 조건 ----------
    st.subheader("추출 조건")
    n_all = len(bd)
    c1, c2, c3 = st.columns(3)
    capacity = c1.number_input("통화 가능 건수", 1, n_all, min(300, n_all), step=10,
                               help="이번에 상담팀이 실제로 걸 수 있는 전화 수")
    agents = c2.number_input("상담원 수", 1, 200, 5,
                             help="추출된 명단을 순위대로 돌아가며 배정해, 상담원마다 점수 분포가 고르게 됩니다.")
    mc_mode = c3.selectbox("월 상담 건수(month_contacts) 기준", ["과거 같은 달 평균", "직접 입력"],
                           help="은행이 정하는 캠페인 강도 값. 모든 고객에게 같은 값이 들어가 순위에는 영향이 작습니다.")
    mc_val = None
    if mc_mode == "직접 입력":
        mc_val = c3.number_input("이번 달 상담 건수", 10, 8000, 1000, step=10)

    c1, c2 = st.columns(2)
    excl_rep = c1.checkbox("이번 캠페인에서 이미 여러 번 연락한 고객 제외")
    rep_n = c1.number_input("제외 기준: campaign ≥", 2, 50, 4, disabled=not excl_rep,
                            help="데이터에서 4회 이상 연락한 고객은 가입률이 평균보다 낮았습니다.")
    only_rec = c2.checkbox("임계값 이상(가입 예측) 고객만 추출",
                           help="켜면 통화 가능 건수보다 적게 뽑힐 수 있습니다.")

    # ---------- 3. 점수 계산 & 정렬 ----------
    X = build_features(bd, mc_val)
    score = model.predict_proba(X)[:, 1]
    out = bd.copy()
    out["가입 점수"] = score
    out["예상 가입률"] = cal(score)
    out["판정"] = np.where(score >= thr, "가입 예측", "-")
    if y_true is not None:
        out["실제 y (체험용)"] = np.where(y_true.values == 1, "yes", "no")
    out = out.sort_values("가입 점수", ascending=False, kind="mergesort").reset_index(drop=True)
    out.insert(0, "전체 순위", np.arange(1, len(out) + 1))

    pool = out
    if excl_rep:
        pool = pool[pool["campaign"] < rep_n]
    if only_rec:
        pool = pool[pool["판정"] == "가입 예측"]
    picked = pool.head(int(capacity)).copy()
    picked.insert(0, "통화 순서", np.arange(1, len(picked) + 1))
    picked.insert(1, "담당 상담원", [f"상담원 {(i % agents) + 1:02d}" for i in range(len(picked))])

    # ---------- 4. 요약 ----------
    st.subheader("추출 결과")
    st.caption(f"{source} · 명단 {n_all:,}명 → 조건 통과 {len(pool):,}명 → 추출 {len(picked):,}명")
    exp_pick = picked["예상 가입률"].sum()
    exp_rand = len(picked) * out["예상 가입률"].mean()
    k = st.columns(4)
    k[0].metric("추출 고객", f"{len(picked):,}명", f"명단의 {len(picked) / n_all:.1%}", delta_color="off")
    k[1].metric("예상 가입자", f"{exp_pick:,.0f}명", f"무작위 대비 +{exp_pick - exp_rand:,.0f}명")
    k[2].metric("추출 고객 예상 가입률", f"{picked['예상 가입률'].mean():.1%}" if len(picked) else "-",
                f"명단 평균 {out['예상 가입률'].mean():.1%}", delta_color="off")
    k[3].metric("무작위 대비 효율", f"{exp_pick / exp_rand:.2f}배" if exp_rand > 0 else "-")

    if y_true is not None and len(picked):
        hit = (picked["실제 y (체험용)"] == "yes").sum()
        tot = (out["실제 y (체험용)"] == "yes").sum()
        st.success(f"**체험 검증:** 테스트 고객 중 실제 가입자 {tot:,}명 가운데 **{hit:,}명({hit / tot:.1%})**이 "
                   f"추출 명단 {len(picked):,}명 안에 들어 있습니다. "
                   f"(실제 적중률 {hit / len(picked):.1%} · 무작위로 뽑았다면 약 {tot / len(out):.1%})")

    # 누적 예상 가입자 곡선 → 통화 건수 정하기
    cum = pool["예상 가입률"].cumsum().values
    x = np.arange(1, len(cum) + 1)
    step = max(1, len(x) // 400)
    fig = go.Figure()
    fig.add_scatter(x=x[::step], y=cum[::step], name="점수 순으로 통화", line=dict(color=C_YES, width=3))
    fig.add_scatter(x=x[::step], y=x[::step] * out["예상 가입률"].mean(), name="무작위로 통화",
                    line=dict(color="gray", dash="dot"))
    fig.add_vline(x=len(picked), line_dash="dash", line_color=C_WARN,
                  annotation_text=f"통화 가능 {len(picked):,}건")
    fig.update_layout(title="통화 건수별 누적 예상 가입자", xaxis_title="통화 건수 (점수 높은 순)",
                      yaxis_title="누적 예상 가입자")
    show(fig, 380)
    st.caption("곡선이 완만해지는 지점부터는 통화를 늘려도 가입자가 크게 늘지 않습니다. 인력 배치 판단에 활용하세요. "
               "예상 가입률은 Validation 데이터로 보정한 값이라 실제 캠페인 환경에 따라 달라질 수 있습니다.")

    # ---------- 5. 명단 ----------
    show_cols = (["통화 순서", "담당 상담원", id_col, "가입 점수", "예상 가입률", "판정"]
                 + (["실제 y (체험용)"] if y_true is not None else [])
                 + ["age", "job", "marital", "balance", "housing", "loan", "contact",
                    "campaign", "previous", "poutcome", "전체 순위"])
    fmt = {"가입 점수": "{:.3f}", "예상 가입률": "{:.1%}", "balance": "{:,}"}
    t1, t2 = st.tabs([f"📞 통화 명단 ({len(picked):,}명)", "👥 상담원별 배정"])
    with t1:
        st.dataframe(picked[show_cols].style.format(fmt), width="stretch", height=440, hide_index=True)
    with t2:
        by_agent = (picked.groupby("담당 상담원")
                    .agg(배정=("통화 순서", "size"), 평균점수=("가입 점수", "mean"),
                         예상가입자=("예상 가입률", "sum")).reset_index())
        table(by_agent, "{:,.2f}", hide_index=True)

    c1, c2 = st.columns(2)
    stamp = pd.Timestamp.now().strftime("%Y%m%d")
    c1.download_button("⬇️ 통화 명단 CSV (추출 고객)", picked[show_cols].to_csv(index=False).encode("utf-8-sig"),
                       f"call_list_{stamp}.csv", "text/csv", type="primary", width="stretch")
    c2.download_button("⬇️ 전체 점수 CSV (명단 전체)", out.to_csv(index=False).encode("utf-8-sig"),
                       f"scored_all_{stamp}.csv", "text/csv", width="stretch")

# ==================================================
# 라우팅
# ==================================================
{
    "📊 데이터 개요": page_overview,
    "🧹 전처리 점검": page_preprocess,
    "🔗 변수 연관성": page_association,
    "🧪 파생변수 근거": page_features,
    "🤖 모델 비교": page_models,
    "🎯 최종 모델": page_final,
    "🧮 가입 예측": page_predict,
    "📞 캠페인 시뮬레이션": page_campaign,
    "📋 콜 리스트 우선순위": page_calllist,
}[page]()