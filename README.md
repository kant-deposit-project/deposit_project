# 은행 마케팅 데이터 기반 정기예금 가입 예측

> 브랜치: `moon_youngheum` · 작성자: 문영흠

은행 전화 마케팅 데이터로 정기예금 가입 여부를 예측하고, **클래스 불균형 처리 전후의 성능 변화와 부작용**을 비교·분석한 프로젝트입니다.

## 최종 결과 (Test)

| 모델 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 베이스라인 (Logistic Regression, 처리 없음) | 0.892 | 0.637 | 0.175 | 0.274 |
| LightGBM (새 피처, 처리 없음) | 0.895 | 0.606 | 0.290 | 0.392 |
| **최종: LightGBM + 캠페인 운영 피처 + RUS 1:3** | **0.883** | **0.500** | **0.523** | **0.511** |

- Validation F1 0.509 → Test F1 0.511로 거의 같아, Validation으로 내린 결정들이 새 데이터에서도 유효함을 확인
- 전체 고객의 **12.3%(1,102명)에게 전화해 551명 가입** → 전화당 가입률 50.0%, 무작위 대비 **약 4.3배**
- 전체 가입 고객의 **52.3%**를 찾음

## 데이터

- 출처: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-full.csv`)
- 45,211건, 17개 칼럼, 가입(yes) 11.7% : 미가입(no) 88.3% (약 1 : 7.5)
- `duration`(통화 시간)은 통화가 끝나야 알 수 있는 정보라 **데이터 누수 방지를 위해 제외**

## 진행 과정 (`project.ipynb`)

| 장 | 내용 | 핵심 결과 |
|---|---|---|
| 1. 데이터 전처리 | 결측치(unknown), 중복, 이상치 처리, Train/Val/Test 분할 | 칼럼별로 다르게 처리, 대체값은 Train에서만 계산 |
| 2. 베이스라인 | Logistic Regression, Decision Tree | Accuracy는 Dummy(88.3%)와 같아 평가 지표로 부적합 |
| 3. 불균형 처리 | Class Weight, RUS, SMOTE × 1:3, 1:2, 1:1 | 강할수록 Recall↑ Precision↓, 1:1은 처리 없음보다 나쁨 |
| 4. 모델링 | 9개 모델 × 4가지 처리 + Stacking (37개 조합) | LightGBM + RUS 1:3이 평균 순위 1위 |
| 5. 추가 실험 | 연관성 높은 칼럼 결합 피처, CatBoost 포함 Stacking | 효과 없음 → 새로운 정보가 필요하다는 결론 |
| 6. 캠페인 운영 피처 | 연락 날짜 복원 → 하루·주별 연락 건수 | **F1 0.472 → 0.509, 처음으로 0.5 돌파** |
| 7. 하이퍼파라미터 튜닝 | RandomizedSearch, 교차검증 | Validation에서 개선 없음 → 기본값 유지 |
| 8. 최종 평가 | Test 1회 평가, 피처 중요도, 오분류 분석 | Test F1 0.511 |

### 1. 데이터 전처리

| 대상 | 처리 | 근거 |
|---|---|---|
| job unknown (0.64%) | 행 삭제 | 비율이 매우 작음 |
| education unknown (4.11%) | 직업별 최빈값으로 대체 | 직업–학력 연관성 Cramér's V 0.557, 대체 정확도 70.5% (전체 최빈값 53.9%) |
| contact unknown (28.8%) | 범주로 유지 | 5·6월에 집중된 기록 누락 |
| poutcome unknown (81.8%) | 범주로 유지 | '이전 접촉 없음'이라는 의미 |
| duration 제거 후 중복 16건 | 분할 전 제거 | 같은 행이 Train·Test에 나뉘는 누수 방지 |
| 이상치 | 유지 (previous 275 한 건만 삭제) | 현실적으로 가능한 극단값이며 가입률 차이가 큰 정보 |

- 분할: Train / Validation / Test = 60 : 20 : 20, 층화 분할 (random_state=42)
- **누수 방지 원칙:** 계산이 필요한 처리(최빈값, 인코딩, 스케일링, 클리핑 기준)는 Train에서만 fit하고, 샘플링은 Train에만 적용

### 평가 기준

가입 고객이 11.7%뿐이라 모든 고객을 미가입으로 찍어도 Accuracy가 88.3%입니다. 또 Accuracy·Precision과 Recall은 서로 반대로 움직여 **지표별 1등이 모두 달랐습니다.** 그래서 Accuracy, Precision, Recall, F1의 순위를 각각 매긴 뒤 **평균 순위**로 조합을 선택했습니다.

### 3. 불균형 처리 결과 (Logistic Regression)

| 처리 | Precision | Recall | F1 | 헛걸음(FP) |
|---|---|---|---|---|
| 처리 없음 | 0.639 | 0.180 | 0.281 | 107 |
| RUS 1:3 | 0.519 | 0.327 | 0.401 | 319 |
| RUS 1:2 | 0.449 | 0.416 | 0.432 | 536 |
| RUS 1:1 | 0.252 | 0.637 | 0.361 | 1,988 |

- **개선:** Recall이 18% → 최대 64%로 상승
- **부작용:** Precision 급락, 헛걸음 통화 약 19배 증가, SMOTE는 PR-AUC까지 하락
- 불균형 처리는 고객을 줄 세우는 능력보다 **'가입'으로 판단하는 기준을 낮추는 효과**가 큼
- Precision을 덜 희생하는 **RUS 1:3**을 이후 단계의 중간 강도로 선택

### 6. 캠페인 운영 피처

5장까지 모든 시도에서 F1이 약 0.47에 머물러, 모델이 아니라 **데이터가 담은 정보량의 한계**로 판단했습니다.

- UCI 설명에 데이터가 날짜순으로 정렬되어 있다고 명시되어 있어, 행 순서로 연도를 복원해 실제 연락 날짜를 만듦
- `daily_calls`(그날 연락한 고객 수), `weekly_calls`(그 주에 연락한 고객 수) 생성
- 하루 연락 건수가 적은 날(229건 이하)의 가입률 **35.2%** vs 다른 날 4.6~7.4% → '소수 선별 연락'과 '대량 연락'의 차이
- Train 5겹 교차검증에서 **5개 fold 모두 개선** (평균 F1 0.464 → 0.505)

### 8. 최종 모델 분석

**피처 중요도 (Permutation Importance, Test)**: weekly_calls > daily_calls > poutcome > contact
- 베이스라인에서 1~2위였던 month가 거의 0으로 하락 → '5월 가입률이 낮다'는 패턴이 계절보다 **대량 연락의 영향**이었을 가능성

**오분류 분석**
- 놓친 가입 고객(FN): 대량 연락일, 이전 캠페인 이력 없음에 집중 → 모델이 '가입 가능성이 낮은 조건'으로 배운 특징을 가진 소수의 가입 고객
- 헛걸음 고객(FP): 맞힌 가입 고객과 특징이 거의 같음 → 현재 데이터만으로는 구분이 어려움

## 한계

- 최종 모델이 캠페인 운영 피처에 크게 의존하므로, 실제 운영에서는 **'계획된 연락 건수'**로 정의해 써야 하며 시기 효과를 일부 담고 있을 수 있음
- 같은 고객이 여러 캠페인에 반복 등장해 Train과 Test에 나뉘어 들어갔을 가능성이 있음
- 통화 비용과 가입 이익이 반영되지 않은 임계값 0.5 기준 결과임

## 실행 방법

```bash
# 패키지 설치 (uv 사용)
uv sync

# 또는 pip
pip install pandas numpy matplotlib scipy scikit-learn imbalanced-learn xgboost lightgbm catboost
```

1. `bank-full.csv`를 `project.ipynb`와 같은 폴더에 둡니다.
2. 노트북을 위에서부터 순서대로 실행합니다 (전체 약 25분, CatBoost가 가장 오래 걸림).
3. 실험 결과는 `experiment_cache.pkl`에 자동 저장되어, 중간에 멈춰도 다시 실행하면 이어서 진행합니다. 처음부터 다시 계산하려면 이 파일을 지우세요.
4. CatBoost가 너무 느리면 2-1 셀의 `CATBOOST_ITERATIONS`를 1000 → 500으로 줄일 수 있습니다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `project.ipynb` | 전체 분석 노트북 (1~8장) |
| `bank-full.csv` | 원본 데이터 |
| `baseline_val.csv`, `baseline_importance.csv` | 베이스라인 결과, 피처 중요도 |
| `imbalance_val.csv` | 불균형 처리 비교 결과 (3장) |
| `modeling_val.csv` | 모델링 결과 (4장) |
| `campaign_features_val.csv` | 캠페인 운영 피처 결과 (6장) |
| `tuning_val.csv` | 튜닝 결과 (7장) |
| `pyproject.toml`, `uv.lock` | 실행 환경 |