#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
통합 데이터 탐색 스크립트 (수정)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

# 결과 저장 디렉토리 확인
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 데이터 로드
def load_data(filepath='integrated_data.csv'):
    """통합 데이터 로드"""
    print(f"데이터 로드 중: {filepath}")
    
    # cp949 인코딩으로 로드
    data = pd.read_csv(filepath, encoding='cp949')
    print(f"인코딩 cp949으로 성공적으로 로드됨")
    
    # 날짜 형식 변환
    data['date'] = pd.to_datetime(data['date'])
    
    print(f"데이터 로드 완료: {data.shape}")
    print("\n컬럼명:")
    for col in data.columns:
        print(f"- {col}")
    
    print("\n데이터 타입:")
    print(data.dtypes)
    
    print("\n처음 5행:")
    print(data.head())
    
    print("\n기본 통계량:")
    print(data.describe())
    
    print("\n결측치 확인:")
    print(data.isnull().sum())
    
    return data

def explore_data(data):
    """데이터 탐색"""
    print("\n데이터 탐색 중...")
    
    # 컬럼명 수정 (영어로 변환)
    column_mapping = {
        '코막힘': 'nasal_congestion',
        '인후염': 'pharyngitis',
        '기관지염': 'bronchitis',
        '알레르기비염': 'allergic_rhinitis',
        '호흡기질환': 'respiratory_disease',
        '감기약': 'cold_medicine',
        '비염약': 'rhinitis_medicine',
        '기침약': 'cough_medicine',
        '콧물약': 'runny_nose_medicine',
        '알레르기약': 'allergy_medicine'
    }
    
    # 컬럼명 변경
    data = data.rename(columns=column_mapping)
    
    print("\n수정된 컬럼명:")
    for col in data.columns:
        print(f"- {col}")
    
    # 데이터 저장
    data.to_csv(os.path.join(results_dir, 'processed_data.csv'), index=False)
    print(f"처리된 데이터 저장 완료: {os.path.join(results_dir, 'processed_data.csv')}")
    
    return data

if __name__ == "__main__":
    # 데이터 로드
    data = load_data()
    
    # 데이터 탐색
    data = explore_data(data)
    
    print("\n데이터 탐색 완료!")

