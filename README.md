# 포르투갈 은행 텔레마케팅: 정기예금 가입 예측

포르투갈 은행의 전화 마케팅 데이터(UCI Bank Marketing, `bank-full`)로 **전화를 걸기 전에** 고객의 정기예금 가입 여부를 예측하는 교육용 머신러닝 연구 프로젝트입니다.

새 모델을 만드는 것보다 **기준 논문의 방법을 재현하고, 근거가 있는 경우에만 확장**하는 것을 목표로 했습니다. 모든 단계는 세 가지로 구분해 기록했습니다.

| 구분 | 의미 |
|---|---|
| **Replication (복제)** | 논문에 명시된 방법을 그대로 재현 |
| **Adaptation (적응)** | 데이터·정보 차이 때문에 수정한 부분 |
| **Extension (확장)** | 논문에 없는, 추가로 시도한 방법 |

**기준 논문:** Dong, Y. (2024). *Potential Customer Prediction of Telecom Marketing based on Machine Learning.* Highlights in Science, Engineering and Technology, 92, 138–145. https://doi.org/10.54097/bbqe4m48

---

## 핵심 결과 요약

모두 같은 테스트 세트(9,043행, 가입자 1,058명, 가입률 11.7%)의 결과입니다. 기준선: 항상 "가입 안 함"으로 예측하면 정확도 0.883.

| 단계 | 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC | KS | Lift @50% |
|---|---|---|---|---|---|---|---|---|
| Round 1 (복제) | Random Forest (balanced) | 0.841 | 0.385 | 0.596 | 0.468 | 0.804 | 0.489 | 1.692 |
| Round 2 (확장) | CatBoost (가중치 4) | 0.875 | 0.470 | 0.519 | 0.493 | 0.806 | 0.490 | 1.662 |
| **Round 3 (확장, 최종 모델)** | **CatBoost (가중치 4) + 경제 지표** | 0.874 | 0.469 | 0.596 | **0.525** | **0.818** | **0.524** | **1.690** |

- **Round 2**는 Round 1보다 F1이 0.468 → 0.493으로 올랐습니다. 개선은 주로 precision/recall 균형(가입 클래스 가중치 4)에서 나왔고, 순위 성능(AUC)은 거의 같습니다.
- **Round 3 (최종 모델)**은 공개 경제 지표 5개를 통화 월 기준으로 연결해 F1을 0.493 → 0.525로 올렸습니다. Precision은 그대로이고 recall이 올랐습니다. 모델은 테스트 **전에** validation과 세 가지 추가 검증으로 정했고, validation과 테스트 결과가 거의 같았습니다(F1 0.526 → 0.525).
- **모든 점수는 같은 31개월 안에서 무작위로 나눈 테스트 결과**입니다. 처음 보는 기간을 예측하면 Round 1~3 모두 이보다 낮을 가능성이 큽니다([Round 3](#round-3--외부-경제-지표-확장) ② 참고).
- **실무 관점 (최종 모델):** 고객 1,345명(14.9%)에게 전화하면 가입자 631명(59.6%)을 찾고, 적중률은 46.9%로 무작위 전화(11.7%)의 약 4배입니다. Round 2보다 176명 더 전화하고 가입자를 82명 더 찾습니다. 정확도(0.874)는 기준선보다 약간 낮은데, 가입자의 60%를 찾는 대신 정확도를 조금 포기한 결과입니다.
- 테스트 세트는 Round 1, 2, 3에서 **총 세 번** 사용했습니다. Round 2와 3은 사후 확장 결과입니다([한계](#한계) 참고).

---

## 데이터

- **파일:** `bank-full.csv` (과제 지정) — 45,211행 × 17열, 타깃 `y` = yes 5,289건(11.7%)
- **결측치:** `isnull()` 기준 0건. 실제 결측은 문자열 `"unknown"`으로 저장됨
- **논문과의 차이 (Adaptation):** Y. Dong (2024)은 `bank-additional-full.csv`(41,188행)를 사용했습니다. 이 파일에는 논문 데이터에 있는 거시경제 지표 5개와 `day_of_week`가 없고, 대신 `balance`와 `day`가 있습니다. 따라서 **논문 점수와 직접 비교할 수 없으며**, 논문 결과는 맥락 참고용으로만 썼습니다. Round 3에서는 비슷한 경제 지표를 공개 통계(Eurostat)에서 직접 만들어 붙였습니다.

### 데이터 분할

- 계층화 분할, `random_state=42`
- **train 60% (27,126) / validation 20% (9,042) / test 20% (9,043)**
- 먼저 80/20으로 나눠 테스트 20%를 고정하고, 남은 80%를 다시 75/25로 나눴습니다. 그래서 Round 1~3의 테스트 고객이 모두 같습니다.
- 결측 대체 규칙, 인코딩, 튜닝은 **train에서만 학습**하고, validation·test에는 적용만 했습니다.
- 테스트 셀은 `RUN_TEST = False`로 잠가 두고, 최종 평가 때만 열었습니다.

---

## 데이터 정제

| 항목 | 처리 | 이유 |
|---|---|---|
| `duration` (통화 시간) | **삭제** | 통화가 끝나야 알 수 있는 값이라 사전 예측에 쓰면 누수(leakage). Y. Dong (2024) Sec. 2와 동일 (Replication) |
| `job`, `education`의 unknown | 규칙 기반 대체 (아래 표) | 논문은 관계를 이용해 추정했다고만 밝힘. 기준은 우리가 정함 (Adaptation) |
| `contact`의 unknown | 범주로 유지 | 5~6월에 몰린 기록 누락 (unknown의 96%) |
| `poutcome`의 unknown | 범주로 유지 | "이전 캠페인 연락 없음"을 뜻함 (`pdays = -1`과 거의 일치 확인) |
| `pdays = -1` | 그대로 유지 | B. Dong et al. (2026)과 동일 |
| 이상치 (`balance` 등), `previous = 275` | 제거하지 않음 | 오류인지 확인할 수 없음 |
| 중복 | 완전 중복 0건. `duration` 제외 중복 16행은 유지 | 고객 ID가 없어 다른 고객일 가능성 |

**결측 대체 규칙** — train에서 "가장 흔한 범주가 맞는 비율(정확도)"을 측정해 결정했습니다.

| 규칙 | 정확도 (train) | 결정 |
|---|---|---|
| 직업 → 교육 수준 (직업별 최빈값) | 0.45–0.86 | 적용 |
| 나이 60세 이상 → `retired` | 0.62 | 적용 |
| 교육 primary → `blue-collar` / tertiary → `management` | 0.55 / 0.59 | 적용 |
| 교육 secondary, 60세 미만 나이대, 주택대출, 개인대출 → 직업 | 0.28 이하 | 미적용 |

적용한 규칙(0.45 이상)과 버린 규칙(0.28 이하) 사이에 뚜렷한 간격이 있습니다. 카이제곱 검정과 Cramér's V(직업 × 교육 0.56, 60세 이상 × 은퇴 0.53)로 관계의 강도도 확인했습니다. 대체 후 남은 unknown: job 178, education 127.

---

## 모델링 방법

| 단계 | 선택 | 구분 |
|---|---|---|
| 인코딩 | 원-핫, 첫 범주 삭제, train에서 학습 (41열) | Adaptation |
| 스케일링 | StandardScaler, LR·KNN에만 (Pipeline 안) | Adaptation |
| 모델 | Logistic Regression, KNN, Decision Tree, Random Forest | Replication |
| 클래스 불균형 | `class_weight="balanced"` (KNN은 옵션 없음) | Extension |
| 검증 | `StratifiedKFold(5, shuffle=True, random_state=42)`. 데이터가 날짜순이라 섞음 | Replication + Adaptation |
| 튜닝 기준 | ROC-AUC (논문은 KNN의 k를 AUC로 선택, Sec. 3.2) | Replication / Adaptation |
| 평가 지표 | Accuracy, AUC, KS, 누적 Lift (논문 Sec. 4) + Precision/Recall/F1 (확장). 논문의 Profit은 비용·수익 수치가 없어 계산하지 않음 | Replication + Extension |
| 외부 변수 (Round 3) | Eurostat 경제 지표 5개를 통화 월 기준으로 연결 | Extension |

**최종 하이퍼파라미터:** LR `C=0.003` · KNN `k=100` · DT `max_depth=10`, `min_samples_leaf=20` · RF 트리 200개, `min_samples_leaf=5` · CatBoost `learning_rate=0.03`, `depth=6`, 500회, 가입 클래스 가중치 4. 최종 모델(Round 3)은 같은 CatBoost 설정에 경제 지표 5열만 추가했습니다.

---

## Round 1 — 논문 복제

| 모델 | Accuracy (CV / test) | AUC (CV / test) | KS (test) | Lift @50% | F1 (test) |
|---|---|---|---|---|---|
| Logistic Regression | 0.763 / 0.762 | 0.764 / 0.773 | 0.414 | 1.620 | 0.380 |
| KNN | 0.892 / 0.893 | 0.760 / 0.767 | 0.440 | 1.584 | 0.286 |
| Decision Tree | 0.802 / 0.814 | 0.768 / 0.779 | 0.453 | 1.595 | 0.430 |
| **Random Forest** | 0.840 / 0.841 | 0.794 / **0.804** | **0.489** | **1.692** | **0.468** |

- AUC·KS 기준 순위가 논문(Tables 5–6)과 같습니다: Random Forest 1위, Decision Tree 2위. LR과 KNN은 순서가 바뀌었지만 차이가 작습니다(0.773 vs 0.767).
- CV와 테스트 점수가 비슷해 과적합은 보이지 않습니다.

---

## Round 2 — 확장 (모든 선택은 train CV / validation에서)

| 단계 | 시도 | 결과 |
|---|---|---|
| A1 | 가입 클래스 가중치 1 / 2 / 4 / 7.5 / 10 / 15 | LR·DT·RF 모두 가중치 4에서 F1 최고. AUC는 거의 변화 없음 |
| A2 | RF `max_features`, KNN 거리·가중, DT `criterion` | KNN 맨해튼 거리만 개선 (80% 분할에서 AUC +0.008, 60% 분할에서는 표준편차 수준) |
| A3 | 오버샘플링, 언더샘플링, SMOTE-NC | 오버샘플링 ≈ 클래스 가중치. SMOTE-NC는 모든 모델에서 AUC 하락 |
| B1–B2 | 날짜 복원 후 `day_of_week`, `year` 추가 | 요일 효과 없음. 연도는 크게 개선되지만 새 연도에 쓸 수 없어 제외 |
| C1 | CatBoost | train CV AUC 0.8005 (RF 0.7943) |
| D1 | 후보 9개를 validation에서 비교 | 1위 CatBoost (가중치 4), 2위 RF (가중치 4) |

A3와 C1은 Round 1 분할의 train 80%에서 실행했습니다(실행 시간이 길어 새 분할에서 다시 돌리지 않음). 이 80%에는 지금의 validation 행이 포함돼 있습니다([한계](#한계) 참고).

**Validation 비교 (상위 4개, cutoff 0.5):**

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **CatBoost (가중치 4)** | 0.875 | 0.471 | 0.546 | **0.506** | **0.811** | 0.468 |
| RF (가중치 4) | 0.876 | 0.474 | 0.509 | 0.490 | 0.800 | **0.469** |
| RF (balanced) | 0.841 | 0.386 | 0.612 | 0.474 | 0.799 | 0.463 |
| CatBoost (balanced) | 0.824 | 0.361 | 0.651 | 0.464 | 0.811 | 0.462 |

Round 2 모델로 CatBoost (가중치 4)를 정한 뒤 train + validation(80%)으로 다시 학습하고 테스트를 한 번 평가했습니다(결과는 맨 위 요약표). 테스트에서는 CatBoost와 RF가 사실상 같은 수준이었습니다(F1 0.493 vs 0.484).

**탐색했지만 쓰지 않은 것 (D2):** validation에서 CatBoost·RF 설정 세부 조정, 두 모델의 평균(앙상블), cutoff 최적화를 시도했습니다. F1 향상이 0.004~0.006 정도였고, 같은 validation에서 고른 값이라 낙관적이므로 채택하지 않았습니다.

---

## Round 3 — 외부 경제 지표 (확장)

**아이디어:** 논문 데이터(`bank-additional`)에는 같은 캠페인에 거시경제 지표 5개가 붙어 있습니다. 우리 파일에는 없으므로, 복원한 통화 날짜(월)를 기준으로 Eurostat 공개 지표를 직접 연결했습니다. 지표는 고객과 같은 표본에서 나온 값이 아니라, **통화 시점의 경제 환경을 나타내는 외부 변수**입니다.

| 지표 | Eurostat 데이터셋 | 지역 | 시차 |
|---|---|---|---|
| 3개월 금리 (월평균) | `irt_st_m` | 유로존 | 1개월 |
| 물가상승률 (HICP, 전년 동월 대비) | `prc_hicp_manr` | 포르투갈 | 1개월 |
| 소비자 신뢰지수 | `ei_bsco_m` | 포르투갈 | 1개월 |
| 실업률 | `une_rt_m` | 포르투갈 | 2개월 |
| 10년 국채 금리 (월평균) | `irt_lt_mcby_m` | 포르투갈 | 1개월 |

시차는 통화 시점에 이미 발표된 값만 쓰기 위한 것입니다(발표 시점은 근사). 통화 월은 31개뿐이라, 지표가 사실상 "몇 월인지"를 알려주는 표시가 될 위험이 있었습니다. 그래서 테스트 전에 세 가지를 확인했습니다. 모두 train + validation만 사용했고, 설정은 Round 2 모델과 같습니다(가중치 4, cutoff 0.5).

**① 무작위 분할 (train 60% → validation 20%, 9,042행)**

| 모델 | 추가 열 | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| CatBoost | 없음 | 0.506 | 0.811 | 0.468 |
| CatBoost | 월 번호만 | 0.523 | 0.820 | 0.494 |
| CatBoost | 경제 지표 5개 | 0.526 | 0.820 | 0.490 |
| Random Forest | 없음 | 0.490 | 0.800 | 0.469 |
| Random Forest | 월 번호만 | 0.515 | 0.812 | 0.490 |
| Random Forest | 경제 지표 5개 | 0.522 | 0.815 | 0.487 |

**② 무작위 5-fold vs 월 단위 분리 5-fold (CatBoost, 36,168행 out-of-fold 예측)**

| 분할 | 추가 열 | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 무작위 (같은 월이 학습·평가에 섞임) | 없음 | 0.489 | 0.801 | 0.445 |
| 무작위 | 월 번호만 | 0.515 | 0.812 | 0.470 |
| 무작위 | 경제 지표 5개 | 0.518 | 0.814 | 0.474 |
| **월 단위 분리** (처음 보는 월만 평가) | 없음 | 0.226 | 0.638 | 0.274 |
| 월 단위 분리 | 월 번호만 | 0.270 | 0.708 | 0.345 |
| 월 단위 분리 | 경제 지표 5개 | **0.289** | **0.735** | **0.367** |

**③ 시간 순서 (2008–09 학습 → 2010 평가, 2,064행)**

| 모델 | 경제 지표 | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| CatBoost | 없음 | 0.641 | 0.635 | 0.653 |
| CatBoost | 있음 | 0.670 | 0.655 | 0.665 |
| Random Forest | 없음 | 0.646 | 0.665 | 0.670 |
| Random Forest | 있음 | 0.662 | 0.642 | 0.652 |

2010년 가입률은 51.4%(2008–09년은 약 9%)라서 ③의 F1은 기준 가입률 때문에 높게 나옵니다. 모두 "가입"이라고 예측해도 F1 ≈ 0.68이므로, ③은 ROC-AUC로만 비교합니다.

**해석:**
- 무작위 분할(①②)에서는 "월 번호"만 넣어도 개선 폭이 경제 지표와 거의 같습니다(① CatBoost F1 +0.017 vs +0.020). 이 조건에서는 경제 정보와 기간 암기를 구분할 수 없습니다. 무작위 분할에서는 같은 월의 나중 통화 결과까지 학습에 들어가기 때문입니다(일종의 시간 누수, Kaufman et al. 2012).
- 처음 보는 월(②)에서는 경제 지표가 "없음"보다 ROC-AUC +0.097, F1 +0.063, "월 번호"보다 ROC-AUC +0.027, F1 +0.019 높았습니다. **지표에 다른 월로 옮겨지는 실제 정보가 있다**는 근거입니다.
- 다만 처음 보는 기간에서는 성능이 크게 떨어집니다(ROC-AUC 0.801 → 0.638, F1 0.489 → 0.226). 무작위 분할 점수는 미래 기간 예측에 낙관적이며, Moro, Cortez & Rita (2014)도 같은 경향을 보고했습니다(시간 순서 평가에서 AUC 0.929 → 0.794).
- 순수한 미래 예측(③)에서는 CatBoost만 좋아졌고(ROC-AUC +0.020), Random Forest는 오히려 떨어졌습니다(−0.023). 2010년 중반부터 국채 금리가 학습 기간 범위를 벗어나 트리 모델이 외삽하지 못했을 가능성이 있지만 검증하지 않았습니다. ③은 2,064행 한 번의 분할이라 ±0.02 차이는 확정적이지 않습니다.
- ②의 월 단위 분리는 평가 월 이후의 데이터로도 학습하므로 미래 예측보다는 "보간"에 가깝습니다(묶음 단위 검증: Roberts et al. 2017).
- 지표는 같은 월의 고객에게 모두 같은 값이라, 효과는 주로 "그 달의 가입 수준"을 맞추는 데서 옵니다. 같은 달 안에서 누구에게 전화할지에 도움이 되는지는 따로 측정하지 않았습니다.
- **`year`는 빼고 경제 지표는 쓴 이유:** 둘 다 기간과 관련 있지만, `year`는 처음 보는 연도에 의미가 없습니다. 경제 지표는 어느 달이든 발표된 값이 있고, 처음 보는 월에서도 효과가 확인됐습니다(②).

**결정:** CatBoost는 ①②③ 모두에서 지표를 넣었을 때 좋아졌으므로(ROC-AUC 기준), **CatBoost (가중치 4) + 경제 지표를 최종 모델**로 정하고 테스트를 한 번 평가했습니다.

**테스트 결과 (한 번 실행):** Y. Dong (2024)과 같은 방식(무작위 분할, 지표를 일반 입력 열로 사용)으로 train + validation(80%)에 다시 학습해 평가했습니다. 수치는 맨 위 요약표의 Round 3 행입니다.
- 지표를 뺀 결과가 Round 2와 정확히 같아(F1 0.493, ROC-AUC 0.806) 실행 과정이 확인됐습니다.
- 지표를 넣으면 F1 0.493 → 0.525, ROC-AUC 0.806 → 0.818, PR-AUC 0.463 → 0.492, KS 0.490 → 0.524였습니다. Precision은 거의 그대로(0.470 → 0.469)이고 recall이 올라, 가입자를 82명 더 찾았습니다(549 → 631명, 연락 대상 1,169 → 1,345명).
- Validation과 테스트가 거의 같습니다(F1 0.526 → 0.525, ROC-AUC 0.820 → 0.818).
- 테스트 세트도 같은 31개월에서 무작위로 뽑았으므로, **절대 점수는 처음 보는 기간에 낙관적**입니다. 이는 Round 1, 2의 점수도 마찬가지입니다.

---

## 변수 해석 (Round 1 모델 기준)

Random Forest의 permutation importance(테스트 AUC 하락폭)와 Logistic Regression 표준화 계수가 같은 요인을 가리킵니다.

| 요인 | 방향 | 주의할 점 |
|---|---|---|
| `contact = unknown` | − | 가장 강하지만 5~6월 기록 누락을 반영. 고객 특성이 아님 |
| `poutcome = success` | + | 이전 캠페인 성공 고객의 가입률 64.7% (전체 11.7%) |
| 주택대출 있음 | − | |
| 이번 캠페인 통화 횟수 (`campaign`) | − | 가입하지 않은 고객에게 다시 전화하는 구조라 인과가 아닌 연관 |
| 월 | ± | 4월 기준 상대값. 원-핫으로 11열로 나뉨 |
| `day`, `age` | RF에서만 중요 | 비선형 효과. RF의 AUC가 더 높은 것과 일치 |

모두 **상관관계이며 인과관계가 아닙니다**. 전화를 덜 하면 가입률이 오른다는 뜻으로 읽으면 안 됩니다. 경제 지표와 가입률의 관계도 마찬가지입니다. 금리가 낮던 시기(2009–10)에 가입률이 높았지만(train에서 Euribor 1.64% 이하 구간 25.0%, 그 위 약 6%), 같은 시기 은행이 통화 수를 크게 줄였기 때문에 경제 때문인지 은행의 고객 선택 때문인지 구분할 수 없습니다.

---

## 한계

- **데이터 버전:** 논문과 파일이 달라 점수를 직접 비교할 수 없습니다. 경제 지표를 추가하자 AUC가 0.806 → 0.818로 올라 논문(0.849)과의 차이가 일부 줄었지만, 행 수와 지표 출처가 달라 남은 차이의 원인은 확인할 수 없습니다.
- **테스트 세트 재사용:** 테스트를 세 번 사용했습니다(Round 1 복제, Round 2, Round 3 최종). Round 2·3의 모델 선택은 테스트 전에 validation과 추가 검증으로 확정했고 테스트 결과를 본 뒤 바꾸지 않았지만, 엄밀한 의미의 "한 번만 사용"은 아니므로 사후 확장 결과로 표시했습니다.
- **기간 일반화:** 무작위 분할 점수(최종 모델 포함, Round 1~3 모두)는 처음 보는 기간에 낙관적입니다(Round 3 ②). Y. Dong (2024)도 80/20 분할이 무작위인지 시간순인지 밝히지 않았습니다.
- **경제 지표:** Eurostat의 현재 수정치를 사용했으며, 당시 실제로 발표된 값(real-time vintage)과 다를 수 있습니다(실업률·소비자 신뢰지수; 시장 금리는 수정되지 않음). 발표 시차는 근사이고, 연도는 파일 순서로 복원했습니다. 순수한 미래 예측(③)에서는 향상이 작았습니다.
- **Round 2 일부 결과의 분할:** A3(resampling)와 C1(CatBoost 튜닝)은 Round 1 분할의 train 80%에서 나와, 이 결정들에 대한 validation 점수는 약간 낙관적일 수 있습니다. 테스트 세트는 이 결정들에 쓰이지 않았습니다.
- **결측 대체:** 직업 → 교육 규칙의 정확도는 약 0.68~0.71로, 대체한 값의 약 30%는 틀렸을 수 있습니다.
- **정확도 기준선:** 최종 모델 정확도(0.874)가 "항상 가입 안 함"(0.883)보다 낮습니다. 불균형 데이터에서는 F1, AUC, Lift로 판단해야 합니다.
- **Profit 미계산:** 논문의 Profit 지표(Sec. 4.5)는 비용·수익 수치가 공개되지 않아 재현하지 않았습니다.

---

## 파일 구성

| 파일 | 내용 |
|---|---|
| `banking project_1.ipynb` | 메인 분석 노트북 (Round 1, 2) |
| `banking project_1 - Copy.ipynb` | Round 3 실험 (경제 지표 연결, 월 단위 검증, 테스트) |
| `banking_project_1_organized_v4.ipynb` | 한국어 설명을 붙여 정리한 최신 버전 (Round 1~3, 최종 정리) |
| `banking_project_1_organized_v3.ipynb` | 이전 정리본 (Round 1, 2) |
| `data-cleaning-standard.md` | 결정 기록 (날짜·근거·결과) |
| `bank_full_with_economic_data.csv` | 고객별로 연결된 경제 지표를 확인하기 위한 파일. 모델은 읽지 않음 |

## 실행 방법

```bash
pip install pandas numpy scikit-learn imbalanced-learn catboost scipy matplotlib truststore
```

1. `bank-full.csv`를 내려받아 노트북 첫 셀의 경로를 수정합니다.
2. 노트북을 위에서 아래로 실행합니다. 각 셀은 위 셀의 결과를 사용합니다.
3. 테스트 셀은 `RUN_TEST = True`로 바꿔야 실행됩니다.
4. Round 3 노트북은 Eurostat API에 접속합니다. Windows에서 인증서 오류가 나면 `truststore`가 Windows 인증서 저장소를 사용하게 합니다(검증을 끄지 않음).

---

## 참고 문헌

- Dong, Y. (2024). Potential Customer Prediction of Telecom Marketing based on Machine Learning. *Highlights in Science, Engineering and Technology*, 92, 138–145. https://doi.org/10.54097/bbqe4m48
- Dong, B. et al. (2026). Risk-Sensitive Machine Learning for Financial Decision Modeling Under Imbalanced Data: Evidence from Bank Telemarketing. *Entropy*, 28(3), 354. https://doi.org/10.3390/e28030354
- Peter, M. et al. (2025). Predicting customer subscription in bank telemarketing campaigns using ensemble learning models. *Machine Learning with Applications*, 19, 100618. https://doi.org/10.1016/j.mlwa.2025.100618
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD*, 6(4). https://doi.org/10.1145/2382577.2382579
- Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Strobl, C. et al. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25. https://doi.org/10.1186/1471-2105-8-25
- Eurostat. 데이터셋 `irt_st_m`, `prc_hicp_manr`, `ei_bsco_m`, `une_rt_m`, `irt_lt_mcby_m` (Eurostat dissemination API, 2026-10-15 접속). https://ec.europa.eu/eurostat
- 데이터: UCI Machine Learning Repository, Bank Marketing (Moro, Laureano & Cortez, 2011)
