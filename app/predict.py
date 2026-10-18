"""
1단계-2: 배치 예측 (캠페인 전 연락 대상 목록 만들기)

실행 예시:
  python predict.py sample_customers.csv --daily 200 --weekly 700
  python predict.py customers.csv                      # 파일에 daily_calls, weekly_calls 칼럼이 있을 때

결과: predictions.csv (전체 고객 + 예측), call_list.csv (가입으로 분류된 고객만)
"""
import argparse
import joblib
import pandas as pd
from deposit_model import predict_customers

parser = argparse.ArgumentParser(description='정기예금 가입 여부 배치 예측')
parser.add_argument('input', help='고객 목록 CSV 파일')
parser.add_argument('--daily', type=int, help='캠페인 계획상 하루 연락 예정 건수 (파일에 칼럼이 없을 때)')
parser.add_argument('--weekly', type=int, help='캠페인 계획상 주간 연락 예정 건수 (파일에 칼럼이 없을 때)')
parser.add_argument('--model', default='model/deposit_model.joblib')
args = parser.parse_args()

artifact = joblib.load(args.model)
customers = pd.read_csv(args.input)
result, warnings = predict_customers(artifact, customers, args.daily, args.weekly)

for w in warnings:
    print('[확인]', w)

result.to_csv('predictions.csv', index=False, encoding='utf-8-sig')
call_list = result[result['예측'] == '가입']
call_list.to_csv('call_list.csv', index=False, encoding='utf-8-sig')

print(f'전체 {len(result):,}명 중 가입으로 분류된 고객 {len(call_list):,}명 ({len(call_list) / len(result):.1%})')
print('저장 완료: predictions.csv, call_list.csv')
