#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
미세먼지 농도와 의약품 소비 데이터 연관성 분석을 위한 탐색적 데이터 분석(EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 결과 저장 디렉토리 확인
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 데이터 로드
def load_processed_data(filepath='results/processed_data.csv'):
    """전처리된 데이터 로드"""
    print(f"전처리된 데이터 로드 중: {filepath}")
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    print(f"데이터 로드 완료: {data.shape}")
    return data

def add_time_features(data):
    """시간 관련 특성 추가"""
    print("\n시간 관련 특성 추가 중...")
    
    # 연도, 월, 일, 요일, 계절 추가
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    
    # 계절 추가 (1: 봄, 2: 여름, 3: 가을, 4: 겨울)
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 4)
    
    # 계절 이름 추가
    season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    data['season_name'] = data['season'].map(season_names)
    
    print("시간 관련 특성 추가 완료")
    return data

def plot_time_series(data):
    """시계열 데이터 시각화"""
    print("\n시계열 데이터 시각화 중...")
    
    # 1. 미세먼지 농도 시계열 그래프
    plt.figure(figsize=(14, 7))
    plt.plot(data['date'], data['pm10'], label='PM10', color='blue', alpha=0.7)
    plt.plot(data['date'], data['pm25'], label='PM2.5', color='red', alpha=0.7)
    plt.title('Fine Dust Concentration Time Series (2023-2024)')
    plt.xlabel('Date')
    plt.ylabel('Concentration (μg/m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fine_dust_time_series.png'))
    plt.close()
    
    # 2. 의약품 검색량 시계열 그래프
    medicine_cols = ['cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'allergy_medicine']
    plt.figure(figsize=(14, 7))
    for col in medicine_cols:
        plt.plot(data['date'], data[col], label=col.replace('_', ' ').title(), alpha=0.7)
    plt.title('Medicine Search Volume Time Series (2023-2024)')
    plt.xlabel('Date')
    plt.ylabel('Search Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'medicine_search_time_series.png'))
    plt.close()
    
    # 3. 질병 검색량 시계열 그래프
    disease_cols = ['nasal_congestion', 'pharyngitis', 'bronchitis', 'allergic_rhinitis', 'respiratory_disease']
    plt.figure(figsize=(14, 7))
    for col in disease_cols:
        plt.plot(data['date'], data[col], label=col.replace('_', ' ').title(), alpha=0.7)
    plt.title('Disease Search Volume Time Series (2023-2024)')
    plt.xlabel('Date')
    plt.ylabel('Search Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'disease_search_time_series.png'))
    plt.close()
    
    # 4. 미세먼지와 주요 의약품/질병 검색량 비교 그래프
    key_vars = ['cold_medicine', 'rhinitis_medicine', 'allergic_rhinitis', 'respiratory_disease']
    
    fig, axes = plt.subplots(len(key_vars), 1, figsize=(14, 12), sharex=True)
    
    for i, var in enumerate(key_vars):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # 미세먼지 농도
        ax1.plot(data['date'], data['pm10'], color='blue', alpha=0.5, label='PM10')
        ax1.plot(data['date'], data['pm25'], color='red', alpha=0.5, label='PM2.5')
        ax1.set_ylabel('Fine Dust (μg/m³)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 검색량
        ax2.plot(data['date'], data[var], color='green', label=var.replace('_', ' ').title())
        ax2.set_ylabel('Search Volume', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        ax1.set_title(f'Fine Dust vs {var.replace("_", " ").title()}')
        ax1.grid(True, alpha=0.3)
        
        # 범례 추가
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fine_dust_vs_search_volume.png'))
    plt.close()
    
    print("시계열 데이터 시각화 완료")

def plot_seasonal_patterns(data):
    """계절적 패턴 시각화"""
    print("\n계절적 패턴 시각화 중...")
    
    # 1. 계절별 미세먼지 농도 boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='season_name', y='pm10', data=data, hue='season_name', palette='Blues', legend=False)
    plt.title('PM10 Concentration by Season')
    plt.xlabel('Season')
    plt.ylabel('PM10 Concentration (μg/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pm10_by_season_boxplot.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='season_name', y='pm25', data=data, hue='season_name', palette='Reds', legend=False)
    plt.title('PM2.5 Concentration by Season')
    plt.xlabel('Season')
    plt.ylabel('PM2.5 Concentration (μg/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pm25_by_season_boxplot.png'))
    plt.close()
    
    # 2. 계절별 의약품 검색량 boxplot
    medicine_cols = ['cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'allergy_medicine']
    plt.figure(figsize=(14, 8))
    
    for i, col in enumerate(medicine_cols, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='season_name', y=col, data=data, hue='season_name', palette='Greens', legend=False)
        plt.title(f'{col.replace("_", " ").title()} by Season')
        plt.xlabel('Season')
        plt.ylabel('Search Volume')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'medicine_search_by_season_boxplot.png'))
    plt.close()
    
    # 3. 계절별 질병 검색량 boxplot
    disease_cols = ['nasal_congestion', 'pharyngitis', 'bronchitis', 'allergic_rhinitis', 'respiratory_disease']
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(disease_cols, 1):
        plt.subplot(3, 2, i)
        sns.boxplot(x='season_name', y=col, data=data, hue='season_name', palette='Oranges', legend=False)
        plt.title(f'{col.replace("_", " ").title()} by Season')
        plt.xlabel('Season')
        plt.ylabel('Search Volume')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'disease_search_by_season_boxplot.png'))
    plt.close()
    
    # 4. 계절별 통계 저장
    seasonal_stats = data.groupby('season_name').agg({
        'pm10': ['mean', 'std', 'min', 'max'],
        'pm25': ['mean', 'std', 'min', 'max'],
        'cold_medicine': ['mean', 'std', 'min', 'max'],
        'rhinitis_medicine': ['mean', 'std', 'min', 'max'],
        'allergy_medicine': ['mean', 'std', 'min', 'max'],
        'respiratory_disease': ['mean', 'std', 'min', 'max']
    })
    
    seasonal_stats.to_csv(os.path.join(results_dir, 'seasonal_statistics.csv'))
    
    print("계절적 패턴 시각화 완료")

def plot_correlation_analysis(data):
    """상관관계 분석 및 시각화"""
    print("\n상관관계 분석 및 시각화 중...")
    
    # 분석에 사용할 변수 선택
    analysis_vars = [
        'pm10', 'pm25', 'no2', 'o3', 'co', 'so2',
        'nasal_congestion', 'pharyngitis', 'bronchitis', 'allergic_rhinitis', 'respiratory_disease',
        'cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'runny_nose_medicine', 'allergy_medicine'
    ]
    
    # 상관관계 계산
    corr_matrix = data[analysis_vars].corr()
    
    # 1. 전체 상관관계 히트맵
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                linewidths=0.5, annot_kws={"size": 8})
    plt.title('Correlation Heatmap of Fine Dust and Health Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 2. 미세먼지와 의약품/질병 검색량 간의 상관관계만 추출
    dust_vars = ['pm10', 'pm25']
    health_vars = [var for var in analysis_vars if var not in ['pm10', 'pm25', 'no2', 'o3', 'co', 'so2']]
    
    dust_health_corr = corr_matrix.loc[dust_vars, health_vars]
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(dust_health_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                linewidths=0.5, annot_kws={"size": 10})
    plt.title('Correlation between Fine Dust and Health Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fine_dust_search_correlation.png'))
    plt.close()
    
    # 3. 산점도 분석
    # PM10과 주요 의약품 검색량 산점도
    medicine_cols = ['cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'allergy_medicine']
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(medicine_cols, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x='pm10', y=col, data=data, alpha=0.6, hue='season_name', palette='viridis')
        
        # 추세선 추가
        sns.regplot(x='pm10', y=col, data=data, scatter=False, color='red')
        
        plt.title(f'PM10 vs {col.replace("_", " ").title()}')
        plt.xlabel('PM10 Concentration (μg/m³)')
        plt.ylabel('Search Volume')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pm10_vs_medicine_scatter.png'))
    plt.close()
    
    # PM10과 주요 질병 검색량 산점도
    disease_cols = ['nasal_congestion', 'pharyngitis', 'allergic_rhinitis', 'respiratory_disease']
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(disease_cols, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x='pm10', y=col, data=data, alpha=0.6, hue='season_name', palette='viridis')
        
        # 추세선 추가
        sns.regplot(x='pm10', y=col, data=data, scatter=False, color='red')
        
        plt.title(f'PM10 vs {col.replace("_", " ").title()}')
        plt.xlabel('PM10 Concentration (μg/m³)')
        plt.ylabel('Search Volume')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pm10_vs_disease_scatter.png'))
    plt.close()
    
    # PM2.5와 주요 의약품 검색량 산점도
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(medicine_cols, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x='pm25', y=col, data=data, alpha=0.6, hue='season_name', palette='viridis')
        
        # 추세선 추가
        sns.regplot(x='pm25', y=col, data=data, scatter=False, color='red')
        
        plt.title(f'PM2.5 vs {col.replace("_", " ").title()}')
        plt.xlabel('PM2.5 Concentration (μg/m³)')
        plt.ylabel('Search Volume')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pm25_vs_medicine_scatter.png'))
    plt.close()
    
    # PM2.5와 주요 질병 검색량 산점도
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(disease_cols, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x='pm25', y=col, data=data, alpha=0.6, hue='season_name', palette='viridis')
        
        # 추세선 추가
        sns.regplot(x='pm25', y=col, data=data, scatter=False, color='red')
        
        plt.title(f'PM2.5 vs {col.replace("_", " ").title()}')
        plt.xlabel('PM2.5 Concentration (μg/m³)')
        plt.ylabel('Search Volume')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pm25_vs_disease_scatter.png'))
    plt.close()
    
    print("상관관계 분석 및 시각화 완료")

def analyze_lag_effects(data):
    """시차 효과 분석"""
    print("\n시차 효과 분석 중...")
    
    # 분석할 변수 선택
    dust_vars = ['pm10', 'pm25']
    health_vars = [
        'nasal_congestion', 'pharyngitis', 'allergic_rhinitis', 'respiratory_disease',
        'cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'allergy_medicine'
    ]
    
    # 시차 범위 설정 (0일부터 7일까지)
    lags = range(8)
    
    # 결과 저장을 위한 데이터프레임 초기화
    lag_corr_results = pd.DataFrame(index=pd.MultiIndex.from_product([dust_vars, health_vars], 
                                                                     names=['dust_var', 'health_var']),
                                    columns=lags)
    
    # 각 변수 쌍에 대해 시차 상관관계 계산
    for dust_var in dust_vars:
        for health_var in health_vars:
            for lag in lags:
                # 시차 적용
                if lag == 0:
                    corr = data[dust_var].corr(data[health_var])
                else:
                    corr = data[dust_var].iloc[:-lag].corr(data[health_var].iloc[lag:])
                
                lag_corr_results.loc[(dust_var, health_var), lag] = corr
    
    # 결과 저장
    lag_corr_results.to_csv(os.path.join(results_dir, 'lag_correlation.csv'))
    
    # 시차 효과 히트맵 시각화
    plt.figure(figsize=(14, 10))
    
    # PM10 시차 효과 히트맵
    plt.subplot(1, 2, 1)
    pm10_data = lag_corr_results.xs('pm10', level='dust_var').T
    sns.heatmap(pm10_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, annot_kws={"size": 8})
    plt.title('Lag Correlation: PM10 vs Health Variables')
    plt.xlabel('Health Variables')
    plt.ylabel('Lag (Days)')
    
    # PM2.5 시차 효과 히트맵
    plt.subplot(1, 2, 2)
    pm25_data = lag_corr_results.xs('pm25', level='dust_var').T
    sns.heatmap(pm25_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, annot_kws={"size": 8})
    plt.title('Lag Correlation: PM2.5 vs Health Variables')
    plt.xlabel('Health Variables')
    plt.ylabel('Lag (Days)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lag_correlation_heatmap.png'))
    plt.close()
    
    # 주요 변수에 대한 시차 효과 그래프
    key_health_vars = ['cold_medicine', 'rhinitis_medicine', 'allergic_rhinitis', 'respiratory_disease']
    
    for health_var in key_health_vars:
        plt.figure(figsize=(10, 6))
        
        for dust_var in dust_vars:
            lag_corrs = lag_corr_results.loc[(dust_var, health_var)].values
            plt.plot(lags, lag_corrs, marker='o', label=dust_var)
        
        plt.title(f'Lag Correlation: Fine Dust vs {health_var.replace("_", " ").title()}')
        plt.xlabel('Lag (Days)')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{health_var}_lag_correlation.png'))
        plt.close()
    
    print("시차 효과 분석 완료")

def exploratory_data_analysis(data):
    """탐색적 데이터 분석 수행"""
    print("\n탐색적 데이터 분석 수행 중...")
    
    # 시간 관련 특성 추가
    data = add_time_features(data)
    
    # 시계열 데이터 시각화
    plot_time_series(data)
    
    # 계절적 패턴 시각화
    plot_seasonal_patterns(data)
    
    # 상관관계 분석 및 시각화
    plot_correlation_analysis(data)
    
    # 시차 효과 분석
    analyze_lag_effects(data)
    
    print("\n탐색적 데이터 분석 완료!")

if __name__ == "__main__":
    # 데이터 로드
    data = load_processed_data()
    
    # 탐색적 데이터 분석 수행
    exploratory_data_analysis(data)
    
    print("\n모든 분석 완료!")

