# 포르투갈 은행 텔레마케팅: 정기예금 가입 예측

포르투갈 은행의 전화 마케팅 데이터(UCI Bank Marketing, `bank-full`)로 **전화를 걸기 전에** 고객의 정기예금 가입 여부를 예측하는 교육용 머신러닝 연구 프로젝트입니다.

새로운 모델을 만드는 것보다 **기준 논문의 방법을 이해하고 재현한 뒤, 근거가 있는 경우에만 확장**하는 것을 목표로 했습니다. 모든 단계는 다음 세 가지로 구분해 기록했습니다.

| 구분 | 의미 |
|---|---|
| **Replication (복제)** | 논문에 명시된 방법을 그대로 재현 |
| **Adaptation (적응)** | 데이터·정보 차이 때문에 수정한 부분 |
| **Extension (확장)** | 논문에 없는, 추가로 시도한 방법 |

- **기준 논문:** Dong, Y. (2024). *Potential Customer Prediction of Telecom Marketing based on Machine Learning.* Highlights in Science, Engineering and Technology, 92, 138–145. https://doi.org/10.54097/bbqe4m48

---

## 핵심 결과 요약

동일한 테스트 세트(9,043행, 가입률 11.7%)에서의 결과입니다. 기준선: 항상 "가입 안 함"으로 예측하면 정확도 0.883.

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC | KS | Lift @ 상위 50% |
|---|---|---|---|---|---|---|---|
| Round 1 최고: Random Forest (balanced) | 0.841 | 0.385 | 0.596 | 0.468 | 0.804 | 0.489 | 1.692 |
| **Round 2 최종: CatBoost (weight 4)** | 0.875 | 0.470 | 0.519 | **0.493** | **0.806** | **0.490** | 1.662 |
| Round 2 차순위: Random Forest (weight 4) | 0.877 | 0.475 | 0.494 | 0.484 | 0.804 | 0.486 | 1.665 |

- Round 2의 개선은 주로 **precision/recall 균형 조정**(가입 클래스 가중치 4)에서 나왔고, 순위 성능(AUC)은 거의 그대로입니다(0.804 → 0.806). CatBoost와 Random Forest는 테스트에서 사실상 동률입니다.
- 정확도는 여전히 "항상 no" 기준선(0.883)보다 약간 낮습니다. 가입자의 약 절반을 찾는 대신 정확도를 조금 포기한 결과입니다.
- 실무 관점(정밀도·재현율에서 역산한 근사치): 고객의 약 13%에게 전화하면 가입자의 약 52%를 찾고, 적중률은 약 47%로 무작위 전화(11.7%)의 약 4배입니다.
- Round 2는 Round 1 이후 테스트 세트를 **한 번 더** 사용한 사후(post-hoc) 확장 결과입니다. 아래 [한계](#한계) 참고.

---

## 데이터

- **파일:** `bank-full.csv` (과제에서 지정) — 45,211행 × 17열 (입력 16 + 타깃 `y`)
- **타깃:** `y` = yes 5,289건(11.70%) / no 39,922건(88.30%)
- **결측치:** `isnull()` 기준 0건. 실제 결측은 문자열 `"unknown"`으로 저장됨
- **논문과의 차이 (Adaptation):** Y. Dong (2024)은 `bank-additional-full.csv`(41,188행, 입력 20개)를 사용했습니다. 이 프로젝트의 파일에는 거시경제 지표 5개(`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`)와 `day_of_week`가 없고, 대신 `balance`와 `day`가 있습니다. 따라서 **논문 점수와 직접 비교할 수 없으며**, 논문 결과는 맥락 참고용으로만 사용했습니다.

---

## 방법

### 1. 데이터 분할 (Adaptation)

- 먼저 80/20 층화 분할(`random_state=42`)로 테스트 세트 20%를 고정합니다.
- 나머지 80%를 다시 75/25로 나눠 **train 60% (27,126행) / validation 20% (9,042행) / test 20% (9,043행)** 로 사용합니다.
- 테스트 세트는 Round 1과 Round 2에서 **동일한 고객**입니다.
- 대체(imputation) 규칙, 인코딩, 튜닝은 모두 train에서만 학습하고 validation·test에는 변환만 적용합니다(누수 방지).
- 테스트 셀은 노트북 끝에 `RUN_TEST = False` 플래그로 잠가 두었습니다.

> Y. Dong (2024)은 80/20 + 5-fold CV를 사용합니다. validation 세트를 따로 둔 것은 Round 2에서 후보 모델을 테스트 세트 없이 비교하기 위한 적응입니다.

### 2. 데이터 정제

| 항목 | 처리 | 근거 / 구분 |
|---|---|---|
| `duration` | **제거** — 통화가 끝나야 알 수 있는 값이라 누수 | Y. Dong (2024), Sec. 2 — Replication |
| `job`, `education`의 `unknown` | 학습 데이터에서 검증한 규칙으로 대체 (아래 표) | Y. Dong Sec. 2를 기준을 명시해 적용 — Adaptation |
| `contact`의 `unknown` | 별도 범주로 유지 (5~6월에 집중된 기록 공백) | 크로스탭으로 확인 |
| `poutcome`의 `unknown` | 별도 범주로 유지 ("이전 캠페인 접촉 없음"을 의미) | `pdays = -1`과 거의 일치 확인 |
| `pdays = -1` | 그대로 유지 | B. Dong et al. (2026), Sec. 2.2.3 |
| 이상치 (`balance` 등) | 제거·윈저화 없음 | B. Dong et al. (2026), Sec. 2.2.1 |
| `previous = 275` (1행) | 유지 (오류 여부 확인 불가) | 이상치 무처리 원칙과 일관 |
| 중복 | 완전 중복 0건. `duration` 제외 중복 16행은 다른 고객일 가능성이 높아 유지 | — |

**`job` / `education` 대체 규칙** — 논문은 "합리적일 때만" 추론했다고만 쓰고 기준을 밝히지 않아, 학습 데이터에서 "가장 흔한 범주가 맞을 확률(정확도)"을 측정해 결정했습니다.

| 규칙 | 정확도 (train) | 결정 |
|---|---|---|
| 직업 → 학력 (직업별 최빈값) | 0.45–0.86 | 적용 |
| 60세 이상 → `retired` | 0.62 | 적용 |
| 학력 primary → `blue-collar` / tertiary → `management` | 0.55 / 0.59 | 적용 |
| 학력 secondary → `blue-collar`, 나이 60세 미만, housing, loan → 직업 | 0.28 이하 | 미적용 |

적용한 규칙(0.45 이상)과 버린 규칙(0.28 이하) 사이에 뚜렷한 간격이 있었고, 카이제곱 검정과 Cramér's V로 관계의 강도도 확인했습니다. 대체 후 남은 `unknown`: job 178, education 127.

### 3. 모델링

| 단계 | 선택 | 구분 |
|---|---|---|
| 인코딩 | One-hot, 첫 범주 제거, train에서 학습 → 41열 | Adaptation |
| 스케일링 | StandardScaler (LR, KNN만, Pipeline 내부) | Adaptation |
| 모델 | Logistic Regression, KNN, Decision Tree, Random Forest | Replication (Y. Dong Sec. 3) |
| 추가 모델 | CatBoost (범주형 변수 직접 처리) | Extension |
| 클래스 불균형 | `class_weight="balanced"`, 이후 가중치 4 비교. KNN은 무작위 오버샘플링 1:2 | Extension |
| 검증 | `StratifiedKFold(5, shuffle=True, random_state=42)` — 데이터가 날짜순이라 셔플 | Replication + Adaptation |
| 튜닝 | 작은 그리드, `scoring="roc_auc"` (KNN의 k를 AUC로 선택한 논문 방식) | Replication / Adaptation |

최종 하이퍼파라미터(train CV 기준): LR `C=0.003`, KNN `k=100`(Round 2에서 Manhattan 거리), DT `max_depth=10, min_samples_leaf=20`, RF `n_estimators=200, min_samples_leaf=5`, CatBoost `learning_rate=0.03, depth=6, iterations=500`.

### 4. 평가 지표

- **Y. Dong (2024)의 지표 (Replication):** Accuracy(다수 클래스 기준선과 함께), ROC-AUC, KS(= max(TPR − FPR)), 누적 Lift (상위 50%)
- **추가 지표 (Extension):** Precision, Recall, F1, PR-AUC — 가입자 11.7%의 불균형 데이터에서 정확도만으로는 성능을 판단하기 어렵기 때문
- 컷오프는 0.5로 고정. 수익(profit)은 논문에 비용·수익 수치가 없어 계산하지 않았습니다.

---

## 결과

### Round 1 — Y. Dong (2024) 복제 (80% 학습, 테스트 1회 사용)

| 모델 | Acc (CV) | Acc (test) | AUC (CV) | AUC (test) | KS (test) | Lift @50% | F1 (test) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.763 | 0.762 | 0.764 | 0.773 | 0.414 | 1.620 | 0.380 |
| KNN | 0.892 | 0.893 | 0.760 | 0.767 | 0.440 | 1.584 | 0.286 |
| Decision Tree | 0.802 | 0.814 | 0.768 | 0.779 | 0.453 | 1.595 | 0.430 |
| **Random Forest** | 0.840 | 0.841 | 0.794 | **0.804** | **0.489** | **1.692** | **0.468** |

- 모델 순위는 논문(Tables 5–6)과 같습니다: AUC·KS 기준 Random Forest 1위, Decision Tree 2위.
- 절댓값은 논문보다 낮습니다(논문 RF AUC 0.849). 데이터 버전 차이, 특히 거시경제 변수 부재와 일관된 결과지만 이를 증명하지는 않습니다.
- 학습 CV와 테스트 점수가 거의 같아 과적합 징후는 없었습니다.
- 같은 `bank-full` 데이터를 쓴 공개 노트북(Gnesi)의 테스트 AUC(LR 0.772, DT 0.753, KNN 0.739)와 비교 가능한 수준입니다.

### Round 2 — 확장 (모든 선택은 train CV / validation에서)

| 단계 | 내용 | 결과 |
|---|---|---|
| A1 | 가입 클래스 가중치 1 / 2 / 4 / ≈7.5(balanced) / 10 / 15 | LR·DT·RF 모두 **가중치 4에서 F1 최고**. AUC는 거의 변화 없음 |
| A2 | 미탐색 하이퍼파라미터 (RF `max_features`, KNN 거리, DT `criterion`) | KNN Manhattan 거리만 의미 있는 개선 |
| A3 | 리샘플링 (오버/언더샘플링, SMOTE-NC) | 오버샘플링 ≈ 클래스 가중치. **SMOTE-NC는 모든 모델에서 AUC 하락** |
| B1–B2 | 날짜 복원(행 순서로 연도 추정) 후 `day_of_week`, `year` 추가 | `day_of_week` 효과 없음. `year`는 성능을 크게 올리지만 미래 연도에 쓸 수 없어 **채택하지 않음** |
| C1 | CatBoost | train CV AUC 0.8005로 RF(0.7943)보다 약간 높음 |
| D1 | validation에서 후보 9개 비교 | 1위 CatBoost (weight 4), 2위 RF (weight 4) |
| E1 | 두 모델을 train+validation(80%)으로 재학습 후 테스트 1회 | [핵심 결과](#핵심-결과-요약) 표 참고 |

### Round 3 — 외부 경제 지표 (확장, 테스트 세트 미사용)

복원한 통화 날짜를 기준으로 Eurostat 월별 지표 5개(3개월 유로 금리, 포르투갈 물가상승률, 소비자 신뢰지수, 실업률, 10년 국채 금리)를 결합했습니다. 각 통화 시점 이전에 발표된 값만 쓰도록 1개월(실업률은 2개월) 시차를 두었습니다.

**월 단위 홀드아웃 검증 (CatBoost w4, train+validation 행만 사용, ROC-AUC):**

| 분할 방식 | 추가 변수 없음 | 월 ID만 | 경제 지표 5개 |
|---|---|---|---|
| 무작위 5-fold (월이 학습·검증에 공유됨) | 0.801 | 0.812 | 0.814 |
| GroupKFold(5), 통화 월 기준 (처음 보는 월) | 0.638 | 0.708 | **0.735** |

- 무작위 분할에서는 경제 지표의 효과와 "어느 달인지 외우는 효과"를 구분할 수 없었습니다(월 ID만으로도 거의 같은 개선).
- 처음 보는 월에서는 경제 지표가 지표 없음보다 +0.097, 월 ID보다 +0.027 AUC 높아, **월을 넘어 전이되는 실제 정보가 있다**는 근거가 됩니다.
- 동시에, 처음 보는 월에서는 모든 모델의 AUC가 크게 떨어집니다(0.801 → 0.638). 무작위 분할로 얻은 점수(이 프로젝트의 E1 포함, 논문도 아마 마찬가지)는 **새로운 기간을 예측할 때 낙관적**입니다. Moro, Cortez & Rita (2014)도 시간 순서 평가에서 비슷한 하락을 보고했습니다.
- 경제 지표를 넣은 모델은 사용자 결정에 따라 **테스트 세트로 평가하지 않았습니다.** 최종 모델은 E1(경제 지표 없음)입니다.

---

## 해석 (Random Forest permutation importance + LR 계수)

두 모델이 공통으로 꼽은 주요 요인:

- **`contact = unknown` (−)** — 가장 강하지만, 5~6월에 몰린 **기록 공백**이라 고객 특성이 아니라 "언제·어떻게 기록됐는지"를 반영합니다. 단독으로 실행 가능한 인사이트는 아닙니다.
- **이전 캠페인 성공 `poutcome = success` (+)** — 가입률 64.7% (전체 11.7%)
- **주택 대출 보유 (−)**, **이번 캠페인 통화 횟수 `campaign` (−)**, **월(month)**
- `campaign`의 음(−) 관계는 "가입하지 않은 고객에게 다시 전화했기 때문"일 수 있어, 통화를 더 하면 가입이 줄어든다는 **인과관계의 증거가 아닙니다.**
- `day`, `age`는 RF에서만 중요 — 직선 모델이 잡지 못하는 비선형 효과로, RF의 더 높은 AUC와 일관됩니다.

---

## 한계

- **데이터 버전 차이:** 논문과 다른 파일이므로 점수 비교는 맥락 참고용입니다.
- **테스트 세트 재사용:** Round 2는 Round 1 결과를 본 뒤 같은 테스트 세트를 한 번 더 사용한 사후 확장입니다. 다만 Round 2의 모든 선택은 validation에서 미리 정했습니다.
- **무작위 분할의 낙관성:** 전체 기간이 31개월에 불과하고 테스트 세트도 같은 기간의 무작위 20%라, 미래 기간 예측 성능은 보고된 점수보다 낮을 가능성이 큽니다(Round 3 참고).
- **정확도:** 최종 모델 정확도(0.875)는 "항상 no" 기준선(0.883)보다 낮습니다.
- **대체 규칙:** CV 내부에서는 대체 매핑을 전체 train으로 학습해 각 fold에 약간 앞서 맞춰져 있습니다(영향은 미미, 테스트는 깨끗함). 대체된 값 중 일부는 틀렸을 가능성이 있습니다(예: 직업→학력 대체 약 30%).
- **기타:** KNN의 k는 유클리드 거리로 튜닝 후 Manhattan으로 재튜닝하지 않음. Round 3의 Eurostat 값은 당시 발표값이 아닌 수정값이며, 월 홀드아웃은 미래 월로도 학습하므로 순수한 예측(forecasting)이 아닌 보간입니다. 모든 실험은 1회 실행(반복 없음).
- **상관 ≠ 인과:** 모든 해석은 연관성이며, 캠페인 대상이 무작위가 아니어서 선택 편향이 있을 수 있습니다.

---

## 실행 방법

### 환경

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn catboost matplotlib
```

Round 3(경제 지표)은 Eurostat API 접속이 필요합니다. Windows에서 SSL 오류가 나면 `truststore`를 설치해 Windows 인증서 저장소를 사용합니다(검증을 끄지 않음).

### 실행 순서

1. 노트북 첫 셀의 데이터 경로를 본인 환경에 맞게 수정합니다.
   ```python
   df = pd.read_csv(r'C:\personal_project_1\bank\bank-full.csv', sep=';', quotechar='"')
   ```
2. 위에서부터 순서대로 실행합니다. 각 셀은 위 셀들이 만든 변수를 사용하므로, 커널을 재시작했다면 "Run All Above" 후 이어서 실행합니다.
3. 테스트 셀은 `RUN_TEST = False`로 잠겨 있습니다. 모든 선택을 마친 뒤 최종 평가 때만 `True`로 바꿉니다.
4. 리샘플링 비교(A3)와 CatBoost 그리드 탐색 셀은 실행 시간이 길어 주석 처리되어 있습니다.

### 파일

| 파일 | 내용 |
|---|---|
| `banking project_1.ipynb` | 분석 노트북 (정제 → 기준선 → 튜닝 → Round 2 확장 → 최종 테스트) |
| `banking_project_1_organized_v3.ipynb` | 정리본 (한국어 설명, Round 1 + Round 2 = Steps 19–27 + 최종 정리) |
| `banking_project_1 - Copy.ipynb` | Round 3 경제 지표 실험 (원본 노트북과 E1 결과는 수정하지 않음) |
| `data-cleaning-standard.md` | 모든 결정·수치·근거를 기록한 작업 문서 |

---

## 참고문헌

- Dong, Y. (2024). Potential Customer Prediction of Telecom Marketing based on Machine Learning. *Highlights in Science, Engineering and Technology*, 92, 138–145. https://doi.org/10.54097/bbqe4m48 — **기준 논문**
- Dong, B. et al. (2026). Risk-Sensitive Machine Learning for Financial Decision Modeling Under Imbalanced Data: Evidence from Bank Telemarketing. *Entropy*, 28(3), 354. https://doi.org/10.3390/e28030354
- Peter, M. et al. (2025). Predicting customer subscription in bank telemarketing campaigns using ensemble learning models. *Machine Learning with Applications*, 19, 100618. https://doi.org/10.1016/j.mlwa.2025.100618
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD*, 6(4). https://doi.org/10.1145/2382577.2382579
- Strobl, C. et al. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25. https://doi.org/10.1186/1471-2105-8-25
- 데이터: UCI Machine Learning Repository, *Bank Marketing* (`bank-full.csv`)