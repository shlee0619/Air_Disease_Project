#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
미세먼지 농도와 의약품 소비 데이터 연관성 분석을 위한 간단한 통계적 모델링
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

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

def build_improved_sarimax_model(data):
    """개선된 SARIMAX 모델 구축 (MinMax 스케일링 및 정상성 변환 적용)"""
    print("\n개선된 SARIMAX 모델 구축 중...")
    
    # 분석할 변수 선택
    target_vars = ['cold_medicine', 'rhinitis_medicine']
    exog_var = 'pm10'  # 외생 변수로 PM10 사용
    
    # 시계열 데이터 준비
    ts_data = data.set_index('date').copy()
    
    # MinMax 스케일링 적용
    scaler = MinMaxScaler()
    ts_data[f'{exog_var}_scaled'] = scaler.fit_transform(ts_data[[exog_var]])
    
    # 예측 기간 설정 (30일)
    forecast_steps = 30
    
    for target_var in target_vars:
        # 정상성 검정
        adf_result = adfuller(ts_data[target_var].dropna())
        is_stationary = adf_result[1] < 0.05
        
        # 변환 적용 여부 초기화
        applied_log_transform = False
        applied_differencing = False
        applied_seasonal_diff = False
        
        # 작업용 데이터 생성
        target_series = ts_data[target_var].copy()
        
        # 1. 분산 안정화 (로그 변환)
        if not is_stationary and (target_series > 0).all():
            log_target = np.log(target_series)
            
            # 로그 변환 결과 시각화
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(target_series)
            plt.title(f'Original {target_var.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(log_target)
            plt.title(f'Log-transformed {target_var.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{target_var}_log_transform.png'))
            plt.close()
            
            # 로그 변환 후 정상성 검정
            adf_log_result = adfuller(log_target.dropna())
            is_stationary_after_log = adf_log_result[1] < 0.05
            
            if is_stationary_after_log or adf_log_result[1] < adf_result[1]:
                target_series = log_target
                applied_log_transform = True
                is_stationary = is_stationary_after_log
        
        # 2. 추세 제거 (차분)
        if not is_stationary:
            diff_target = target_series.diff().dropna()
            
            # 차분 결과 시각화
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(target_series)
            plt.title(f'{"Log-transformed" if applied_log_transform else "Original"} {target_var.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(diff_target)
            plt.title(f'Differenced {target_var.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{target_var}_differencing.png'))
            plt.close()
            
            # 차분 후 정상성 검정
            adf_diff_result = adfuller(diff_target.dropna())
            is_stationary_after_diff = adf_diff_result[1] < 0.05
            
            if is_stationary_after_diff:
                target_series = diff_target
                applied_differencing = True
                is_stationary = True
        
        # 3. 계절성 제거 (계절 차분)
        if not is_stationary:
            seasonal_diff_target = target_series.diff(12).dropna()
            
            # 계절 차분 결과 시각화
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(target_series)
            plt.title(f'{"Log-transformed" if applied_log_transform else "Original"} {target_var.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(seasonal_diff_target)
            plt.title(f'Seasonally Differenced {target_var.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{target_var}_seasonal_diff.png'))
            plt.close()
            
            # 계절 차분 후 정상성 검정
            adf_seasonal_diff_result = adfuller(seasonal_diff_target.dropna())
            is_stationary_after_seasonal_diff = adf_seasonal_diff_result[1] < 0.05
            
            if is_stationary_after_seasonal_diff:
                target_series = seasonal_diff_target
                applied_seasonal_diff = True
                is_stationary = True
        
        # 변환된 시계열 데이터 준비
        if applied_log_transform or applied_differencing or applied_seasonal_diff:
            # 변환된 시계열 데이터 생성
            transformed_data = pd.DataFrame(index=target_series.index)
            transformed_data[f'{target_var}_transformed'] = target_series
            transformed_data[f'{exog_var}_scaled'] = ts_data.loc[target_series.index, f'{exog_var}_scaled']
            
            # 결측치 제거
            transformed_data = transformed_data.dropna()
            
            # 훈련 데이터와 테스트 데이터 분리
            train_size = int(len(transformed_data) * 0.8)
            train_data = transformed_data.iloc[:train_size]
            test_data = transformed_data.iloc[train_size:]
        else:
            # 훈련 데이터와 테스트 데이터 분리
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data.iloc[:train_size]
            test_data = ts_data.iloc[train_size:]
        
        # SARIMAX 모델 파라미터 설정
        # 차분 적용 여부에 따라 d 값 조정
        d = 0 if applied_differencing else 1
        # 계절 차분 적용 여부에 따라 D 값 조정
        D = 0 if applied_seasonal_diff else 1
        
        order = (1, d, 1)  # (p, d, q)
        seasonal_order = (1, D, 1, 12)  # (P, D, Q, s)
        
        try:
            # SARIMAX 모델 구축
            if applied_log_transform or applied_differencing or applied_seasonal_diff:
                model = SARIMAX(
                    train_data[f'{target_var}_transformed'],
                    exog=train_data[[f'{exog_var}_scaled']],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = SARIMAX(
                    train_data[target_var],
                    exog=train_data[[f'{exog_var}_scaled']],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            # 모델 학습
            results = model.fit(disp=False)
            
            # 모델 요약 저장
            with open(os.path.join(results_dir, f'{target_var}_improved_sarimax_summary.txt'), 'w') as f:
                f.write(str(results.summary()))
            
            # 테스트 데이터에 대한 예측
            if applied_log_transform or applied_differencing or applied_seasonal_diff:
                pred = results.get_forecast(steps=len(test_data), exog=test_data[[f'{exog_var}_scaled']])
                pred_ci = pred.conf_int()
                
                # 예측 결과 시각화
                plt.figure(figsize=(12, 6))
                
                # 훈련 데이터
                plt.plot(train_data.index, train_data[f'{target_var}_transformed'], label='Training Data')
                
                # 테스트 데이터
                plt.plot(test_data.index, test_data[f'{target_var}_transformed'], label='Test Data')
                
                # 예측 결과
                plt.plot(test_data.index, pred.predicted_mean, label='Forecast', color='red')
                
                # 신뢰 구간
                plt.fill_between(test_data.index,
                                pred_ci.iloc[:, 0],
                                pred_ci.iloc[:, 1], color='pink', alpha=0.3)
            else:
                pred = results.get_forecast(steps=len(test_data), exog=test_data[[f'{exog_var}_scaled']])
                pred_ci = pred.conf_int()
                
                # 예측 결과 시각화
                plt.figure(figsize=(12, 6))
                
                # 훈련 데이터
                plt.plot(train_data.index, train_data[target_var], label='Training Data')
                
                # 테스트 데이터
                plt.plot(test_data.index, test_data[target_var], label='Test Data')
                
                # 예측 결과
                plt.plot(test_data.index, pred.predicted_mean, label='Forecast', color='red')
                
                # 신뢰 구간
                plt.fill_between(test_data.index,
                                pred_ci.iloc[:, 0],
                                pred_ci.iloc[:, 1], color='pink', alpha=0.3)
            
            # 그래프 설정
            plt.title(f'Improved SARIMAX Forecast: {target_var.replace("_", " ").title()}')
            plt.xlabel('Date')
            plt.ylabel('Search Volume')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{target_var}_improved_sarimax_forecast.png'))
            plt.close()
            
            # 모델 성능 평가
            if applied_log_transform or applied_differencing or applied_seasonal_diff:
                mse = ((pred.predicted_mean - test_data[f'{target_var}_transformed']) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(pred.predicted_mean - test_data[f'{target_var}_transformed']).mean()
                r_squared = 1 - (np.sum((test_data[f'{target_var}_transformed'] - pred.predicted_mean) ** 2) / 
                                np.sum((test_data[f'{target_var}_transformed'] - test_data[f'{target_var}_transformed'].mean()) ** 2))
            else:
                mse = ((pred.predicted_mean - test_data[target_var]) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(pred.predicted_mean - test_data[target_var]).mean()
                r_squared = 1 - (np.sum((test_data[target_var] - pred.predicted_mean) ** 2) / 
                                np.sum((test_data[target_var] - test_data[target_var].mean()) ** 2))
            
            # 결과 저장
            with open(os.path.join(results_dir, f'{target_var}_improved_sarimax_performance.txt'), 'w') as f:
                f.write(f"# {target_var.replace('_', ' ').title()} 개선된 SARIMAX 모델 성능\n\n")
                f.write(f"- 적용된 변환:\n")
                f.write(f"  - 로그 변환: {'적용' if applied_log_transform else '미적용'}\n")
                f.write(f"  - 차분: {'적용' if applied_differencing else '미적용'}\n")
                f.write(f"  - 계절성 차분: {'적용' if applied_seasonal_diff else '미적용'}\n\n")
                f.write(f"- SARIMAX 파라미터:\n")
                f.write(f"  - order: {order}\n")
                f.write(f"  - seasonal_order: {seasonal_order}\n\n")
                f.write(f"- 성능 지표:\n")
                f.write(f"  - MSE: {mse:.4f}\n")
                f.write(f"  - RMSE: {rmse:.4f}\n")
                f.write(f"  - MAE: {mae:.4f}\n")
                f.write(f"  - R²: {r_squared:.4f}\n")
        
        except Exception as e:
            print(f"Error in SARIMAX modeling for {target_var}: {e}")
    
    print("개선된 SARIMAX 모델 구축 완료")

if __name__ == "__main__":
    # 데이터 로드
    data = load_processed_data()
    
    # 개선된 SARIMAX 모델 구축
    build_improved_sarimax_model(data)
    
    print("\n모든 분석 완료!")

