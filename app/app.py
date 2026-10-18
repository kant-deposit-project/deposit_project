"""
2단계: 정기예금 연락 대상 선정 웹앱 (Streamlit)
실행 (프로젝트 폴더에서): streamlit run app/app.py
"""
import os
import runpy
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from deposit_model import predict_customers, BASE_COLS

st.set_page_config(page_title='정기예금 연락 대상 선정', page_icon='🏦', layout='wide')


BASE = Path(__file__).resolve().parent          # app 폴더
MODEL_PATH = BASE / 'model' / 'deposit_model.joblib'


@st.cache_resource
def load_model():
    # 모델 파일이 없으면(처음 배포했을 때 등) 그 자리에서 학습해 저장 (LightGBM이라 수 초면 끝남)
    if not os.path.exists(MODEL_PATH):
        runpy.run_path(str(BASE / 'train_model.py'))
    return joblib.load(MODEL_PATH)


artifact = load_model()

st.title('정기예금 연락 대상 선정')
st.caption('고객 정보와 캠페인 계획을 넣으면, 정기예금 가입이 예상되는 고객을 분류합니다. (예금술사 · DS 2팀)')

# ── 사이드바: 데이터와 캠페인 계획 ─────────────────────────────────
with st.sidebar:
    st.header('1. 고객 데이터')
    uploaded = st.file_uploader('고객 목록 CSV', type='csv')
    use_sample = st.checkbox('시연용 데이터 사용 (Test 고객 500명)', value=uploaded is None)
    st.header('2. 캠페인 계획')
    daily = st.number_input('하루 연락 예정 건수', min_value=1, max_value=2000, value=200, step=10)
    weekly = st.number_input('주간 연락 예정 건수', min_value=1, max_value=10000, value=700, step=50)
    st.caption('파일에 daily_calls, weekly_calls 칼럼이 있으면 파일 값을 사용합니다.')

if uploaded is not None:
    customers = pd.read_csv(uploaded)
elif use_sample:
    customers = pd.read_csv(BASE / 'sample_customers.csv')
else:
    st.info('왼쪽에서 고객 목록 CSV를 올리거나 시연용 데이터를 선택하세요.')
    st.stop()

missing = [c for c in BASE_COLS if c not in customers.columns]
if missing:
    st.error(f'필수 칼럼이 없습니다: {", ".join(missing)}')
    st.stop()

tab_predict, tab_info = st.tabs(['연락 대상 예측', '모델 정보'])

# ── 탭 1: 연락 대상 예측 ───────────────────────────────────────────
with tab_predict:
    result, warnings = predict_customers(artifact, customers, daily, weekly)
    for w in warnings:
        st.warning(w)

    n_total = len(result)
    n_call = int((result['예측'] == '가입').sum())
    c1, c2, c3 = st.columns(3)
    c1.metric('전체 고객', f'{n_total:,}명')
    c2.metric('가입으로 분류된 고객', f'{n_call:,}명')
    c3.metric('연락 대상 비율', f'{n_call / n_total:.1%}')

    only_call = st.toggle('가입으로 분류된 고객만 보기', value=True)
    view = result[result['예측'] == '가입'] if only_call else result
    st.dataframe(view, width='stretch', hide_index=True)

    st.download_button('연락 대상 목록 다운로드 (CSV)',
                       result[result['예측'] == '가입'].to_csv(index=False).encode('utf-8-sig'),
                       file_name='call_list.csv', mime='text/csv')
    st.caption('예측 결과는 우선순위 참고용입니다. Test 기준 실제 가입 고객의 약 48%는 미가입으로 분류되므로, 연락 제외 기준으로 쓰지 마세요.')

# ── 탭 2: 모델 정보 ───────────────────────────────────────────────
with tab_info:
    m = artifact['test_metrics']
    st.subheader('최종 모델')
    st.write('LightGBM (기본 하이퍼파라미터) + 캠페인 운영 피처(daily_calls, weekly_calls) + Random Under Sampling 1:3, 임계값 0.5')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Accuracy', f"{m['Accuracy']:.3f}")
    c2.metric('Precision', f"{m['Precision']:.3f}")
    c3.metric('Recall', f"{m['Recall']:.3f}")
    c4.metric('F1', f"{m['F1']:.3f}")
    st.caption(f"Test 데이터 기준 · 데이터의 실제 가입 비율 {artifact['base_rate']:.1%} · 모델 버전 {artifact['version']}")
    st.markdown('**입력 칼럼**: ' + ', '.join(BASE_COLS) + ' (+ daily_calls, weekly_calls 또는 캠페인 계획값)')
    st.markdown('**한계**: 2008~2010년 포르투갈 은행 데이터로 학습한 모델입니다. 실제 사용하려면 각 은행의 데이터로 다시 학습해야 하며, '
                '통화 비용과 가입 이익은 반영하지 않았습니다.')