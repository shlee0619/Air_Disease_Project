#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
통합 데이터 탐색 스크립트
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
    
    # 다양한 인코딩 시도
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    
    for encoding in encodings:
        try:
            data = pd.read_csv(filepath, encoding=encoding)
            print(f"인코딩 {encoding}으로 성공적으로 로드됨")
            break
        except UnicodeDecodeError:
            print(f"인코딩 {encoding}으로 로드 실패")
    
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
    
    # 컬럼명 수정 (한글 깨짐 문제 해결)
    column_mapping = {
        '�ڸ���': 'cold_medicine',
        '���Ŀ�': 'nasal_congestion',
        '�������': 'pharyngitis',
        '�˷������': 'allergy_medicine',
        'ȣ������ȯ': 'respiratory_disease',
        '�����': 'rhinitis',
        '�񿰾�': 'rhinitis_medicine',
        '��ħ��': 'cough_medicine',
        '�๰��': 'antibiotic',
        '�˷������': 'allergy'
    }
    
    # 컬럼명이 매핑에 있으면 변경, 없으면 그대로 유지
    data.columns = [column_mapping.get(col, col) for col in data.columns]
    
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

