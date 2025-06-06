#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
미세먼지 농도와 의약품 소비 데이터 연관성 분석을 위한 통계적 모델링 (수정)
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

def perform_regression_analysis(data):
    """회귀 분석 수행"""
    print("\n회귀 분석 수행 중...")
    
    # 분석할 변수 선택
    dust_vars = ['pm10', 'pm25']
    health_vars = [
        'cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'allergy_medicine',
        'nasal_congestion', 'pharyngitis', 'allergic_rhinitis', 'respiratory_disease'
    ]
    
    # 결과 저장을 위한 데이터프레임 초기화
    regression_results = pd.DataFrame(columns=[
        'dust_var', 'health_var', 'coef', 'std_err', 'p_value', 'r_squared', 'adj_r_squared'
    ])
    
    # 회귀 분석 결과 요약 텍스트 파일 초기화
    summary_file = os.path.join(results_dir, 'regression_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("# 미세먼지 농도와 의약품/질병 검색량 간의 회귀 분석 결과\n\n")
    
    # 각 변수 쌍에 대해 회귀 분석 수행
    for dust_var in dust_vars:
        for health_var in health_vars:
            # 회귀 분석 모델 구축
            X = sm.add_constant(data[dust_var])
            y = data[health_var]
            model = sm.OLS(y, X).fit()
            
            # 결과 저장
            regression_results = regression_results._append({
                'dust_var': dust_var,
                'health_var': health_var,
                'coef': model.params[dust_var],
                'std_err': model.bse[dust_var],
                'p_value': model.pvalues[dust_var],
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj
            }, ignore_index=True)
            
            # 결과 요약 텍스트 파일에 추가
            with open(summary_file, 'a') as f:
                f.write(f"## {dust_var.upper()} vs {health_var.replace('_', ' ').title()}\n\n")
                f.write(f"- 계수: {model.params[dust_var]:.4f}\n")
                f.write(f"- 표준 오차: {model.bse[dust_var]:.4f}\n")
                f.write(f"- p-value: {model.pvalues[dust_var]:.4f}\n")
                f.write(f"- R²: {model.rsquared:.4f}\n")
                f.write(f"- 조정된 R²: {model.rsquared_adj:.4f}\n\n")
            
            # 회귀 분석 시각화
            plt.figure(figsize=(10, 6))
            plt.scatter(data[dust_var], data[health_var], alpha=0.5)
            
            # 회귀선 추가
            x_range = np.linspace(data[dust_var].min(), data[dust_var].max(), 100)
            y_pred = model.params[0] + model.params[1] * x_range
            plt.plot(x_range, y_pred, color='red', linewidth=2)
            
            # 그래프 설정
            plt.title(f'Simple Regression: {dust_var.upper()} vs {health_var.replace("_", " ").title()}')
            plt.xlabel(f'{dust_var.upper()} Concentration (μg/m³)')
            plt.ylabel('Search Volume')
            plt.grid(True, alpha=0.3)
            
            # 회귀 결과 텍스트 추가
            text = f"y = {model.params[0]:.2f} + {model.params[1]:.2f}x\n"
            text += f"R² = {model.rsquared:.2f}\n"
            text += f"p-value = {model.pvalues[dust_var]:.4f}"
            plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                        va='top', ha='left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{dust_var}_vs_{health_var}_regression.png'))
            plt.close()
    
    # 결과 저장
    regression_results.to_csv(os.path.join(results_dir, 'regression_results.csv'), index=False)
    
    print("회귀 분석 완료")
    return regression_results

def analyze_lag_regression(data):
    """시차 효과를 고려한 회귀 분석"""
    print("\n시차 효과를 고려한 회귀 분석 중...")
    
    # 분석할 변수 선택
    dust_vars = ['pm10', 'pm25']
    health_vars = [
        'cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'allergy_medicine',
        'nasal_congestion', 'pharyngitis', 'allergic_rhinitis', 'respiratory_disease'
    ]
    
    # 결과 저장을 위한 데이터프레임 초기화
    optimal_lag_results = pd.DataFrame(columns=[
        'dust_var', 'health_var', 'optimal_lag', 'coef', 'p_value', 'r_squared'
    ])
    
    # 결과 요약 텍스트 파일 초기화
    summary_file = os.path.join(results_dir, 'lag_regression_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("# 시차 효과를 고려한 회귀 분석 결과\n\n")
    
    # 각 변수 쌍에 대해 시차 효과를 고려한 회귀 분석 수행
    for dust_var in dust_vars:
        for health_var in health_vars:
            # 시차 범위 설정 (0일부터 7일까지)
            lags = range(8)
            
            # 각 시차에 대한 R² 값 저장
            r_squared_values = []
            models = []
            
            for lag in lags:
                # 시차 적용
                if lag == 0:
                    # 시차가 없는 경우 (기본 회귀 분석)
                    X = sm.add_constant(data[dust_var])
                    y = data[health_var]
                else:
                    # 시차가 있는 경우
                    # 데이터프레임 복사 및 시차 적용
                    lagged_data = data.copy()
                    lagged_data[f'{health_var}_lagged'] = lagged_data[health_var].shift(-lag)
                    
                    # 결측치 제거
                    lagged_data = lagged_data.dropna(subset=[f'{health_var}_lagged'])
                    
                    X = sm.add_constant(lagged_data[dust_var])
                    y = lagged_data[f'{health_var}_lagged']
                
                # 회귀 분석 모델 구축
                model = sm.OLS(y, X).fit()
                r_squared_values.append(model.rsquared)
                models.append(model)
            
            # 최적 시차 선택 (R² 값이 가장 높은 시차)
            optimal_lag = lags[np.argmax(r_squared_values)]
            optimal_model = models[optimal_lag]
            
            # 결과 저장
            optimal_lag_results = optimal_lag_results._append({
                'dust_var': dust_var,
                'health_var': health_var,
                'optimal_lag': optimal_lag,
                'coef': optimal_model.params[dust_var],
                'p_value': optimal_model.pvalues[dust_var],
                'r_squared': optimal_model.rsquared
            }, ignore_index=True)
            
            # 결과 요약 텍스트 파일에 추가
            with open(summary_file, 'a') as f:
                f.write(f"## {dust_var.upper()} vs {health_var.replace('_', ' ').title()}\n\n")
                f.write(f"- 최적 시차: {optimal_lag}일\n")
                f.write(f"- 계수: {optimal_model.params[dust_var]:.4f}\n")
                f.write(f"- p-value: {optimal_model.pvalues[dust_var]:.4f}\n")
                f.write(f"- R²: {optimal_model.rsquared:.4f}\n\n")
            
            # 시차별 R² 값 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(lags, r_squared_values, marker='o', linestyle='-')
            plt.axvline(x=optimal_lag, color='red', linestyle='--', alpha=0.7)
            plt.title(f'R² by Lag: {dust_var.upper()} vs {health_var.replace("_", " ").title()}')
            plt.xlabel('Lag (Days)')
            plt.ylabel('R²')
            plt.grid(True, alpha=0.3)
            plt.annotate(f'Optimal Lag: {optimal_lag} days\nR² = {max(r_squared_values):.4f}',
                        xy=(optimal_lag, max(r_squared_values)),
                        xytext=(optimal_lag + 0.5, max(r_squared_values)),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{dust_var}_vs_{health_var}_lag_r_squared.png'))
            plt.close()
    
    # 결과 저장
    optimal_lag_results.to_csv(os.path.join(results_dir, 'optimal_lag_results.csv'), index=False)
    
    print("시차 효과를 고려한 회귀 분석 완료")
    return optimal_lag_results

def analyze_seasonal_regression(data):
    """계절성을 고려한 회귀 분석"""
    print("\n계절성을 고려한 회귀 분석 중...")
    
    # 계절 더미 변수 추가
    data['month'] = data['date'].dt.month
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 4)
    
    # 계절 더미 변수 생성
    season_dummies = pd.get_dummies(data['season'], prefix='season', drop_first=True)
    data = pd.concat([data, season_dummies], axis=1)
    
    # 분석할 변수 선택
    dust_vars = ['pm10', 'pm25']
    health_vars = [
        'cold_medicine', 'rhinitis_medicine', 'cough_medicine', 'allergy_medicine',
        'nasal_congestion', 'pharyngitis', 'allergic_rhinitis', 'respiratory_disease'
    ]
    
    # 결과 저장을 위한 데이터프레임 초기화
    seasonal_regression_results = pd.DataFrame(columns=[
        'dust_var', 'health_var', 'dust_coef', 'dust_p_value', 'r_squared', 'adj_r_squared'
    ])
    
    # 결과 요약 텍스트 파일 초기화
    summary_file = os.path.join(results_dir, 'seasonal_regression_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("# 계절성을 고려한 회귀 분석 결과\n\n")
    
    # 각 변수 쌍에 대해 계절성을 고려한 회귀 분석 수행
    for dust_var in dust_vars:
        for health_var in health_vars:
            # 회귀 분석 모델 구축 (계절 더미 변수 포함)
            X = sm.add_constant(pd.concat([data[[dust_var]], data[['season_2', 'season_3', 'season_4']]], axis=1))
            y = data[health_var]
            model = sm.OLS(y, X).fit()
            
            # 결과 저장
            seasonal_regression_results = seasonal_regression_results._append({
                'dust_var': dust_var,
                'health_var': health_var,
                'dust_coef': model.params[dust_var],
                'dust_p_value': model.pvalues[dust_var],
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj
            }, ignore_index=True)
            
            # 결과 요약 텍스트 파일에 추가
            with open(summary_file, 'a') as f:
                f.write(f"## {dust_var.upper()} vs {health_var.replace('_', ' ').title()}\n\n")
                f.write(f"- {dust_var} 계수: {model.params[dust_var]:.4f} (p-value: {model.pvalues[dust_var]:.4f})\n")
                f.write(f"- 여름 계수: {model.params['season_2']:.4f} (p-value: {model.pvalues['season_2']:.4f})\n")
                f.write(f"- 가을 계수: {model.params['season_3']:.4f} (p-value: {model.pvalues['season_3']:.4f})\n")
                f.write(f"- 겨울 계수: {model.params['season_4']:.4f} (p-value: {model.pvalues['season_4']:.4f})\n")
                f.write(f"- R²: {model.rsquared:.4f}\n")
                f.write(f"- 조정된 R²: {model.rsquared_adj:.4f}\n\n")
    
    # 결과 저장
    seasonal_regression_results.to_csv(os.path.join(results_dir, 'seasonal_regression_results.csv'), index=False)
    
    print("계절성을 고려한 회귀 분석 완료")
    return seasonal_regression_results

def perform_time_series_decomposition(data):
    """시계열 분해 분석"""
    print("\n시계열 분해 분석 중...")
    
    # 분석할 변수 선택
    health_vars = ['cold_medicine', 'rhinitis_medicine', 'allergic_rhinitis', 'respiratory_disease']
    
    # 시계열 데이터 준비
    ts_data = data.set_index('date')
    
    for var in health_vars:
        # 시계열 분해
        decomposition = seasonal_decompose(ts_data[var], model='additive', period=30)
        
        # 시각화
        plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(decomposition.observed)
        plt.title(f'{var.replace("_", " ").title()} - Original')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 2)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 3)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonality')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(4, 1, 4)
        plt.plot(decomposition.resid)
        plt.title('Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{var}_time_series_decomposition.png'))
        plt.close()
    
    print("시계열 분해 분석 완료")

def check_stationarity(data):
    """시계열 정상성 검정"""
    print("\n시계열 정상성 검정 중...")
    
    # 분석할 변수 선택
    vars_to_check = ['pm10', 'pm25', 'cold_medicine', 'rhinitis_medicine', 'allergic_rhinitis', 'respiratory_disease']
    
    # 결과 저장을 위한 데이터프레임 초기화
    stationarity_results = pd.DataFrame(columns=[
        'variable', 'adf_statistic', 'p_value', 'is_stationary'
    ])
    
    # 결과 요약 텍스트 파일 초기화
    summary_file = os.path.join(results_dir, 'stationarity_test_results.txt')
    with open(summary_file, 'w') as f:
        f.write("# 시계열 정상성 검정 결과 (ADF 테스트)\n\n")
    
    # 각 변수에 대해 ADF 테스트 수행
    for var in vars_to_check:
        # ADF 테스트
        result = adfuller(data[var].dropna())
        
        # 결과 저장
        stationarity_results = stationarity_results._append({
            'variable': var,
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }, ignore_index=True)
        
        # 결과 요약 텍스트 파일에 추가
        with open(summary_file, 'a') as f:
            f.write(f"## {var.replace('_', ' ').title()}\n\n")
            f.write(f"- ADF 통계량: {result[0]:.4f}\n")
            f.write(f"- p-value: {result[1]:.4f}\n")
            f.write(f"- 정상성 여부: {'정상' if result[1] < 0.05 else '비정상'}\n\n")
    
    # 결과 저장
    stationarity_results.to_csv(os.path.join(results_dir, 'stationarity_results.csv'), index=False)
    
    print("시계열 정상성 검정 완료")
    return stationarity_results

def build_sarimax_model(data):
    """SARIMAX 모델 구축"""
    print("\nSARIMAX 모델 구축 중...")
    
    # 분석할 변수 선택
    target_vars = ['cold_medicine', 'rhinitis_medicine']
    exog_var = 'pm10'  # 외생 변수로 PM10 사용
    
    # 시계열 데이터 준비
    ts_data = data.set_index('date')
    
    # 예측 기간 설정 (30일)
    forecast_steps = 30
    
    for target_var in target_vars:
        # 훈련 데이터와 테스트 데이터 분리
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data.iloc[:train_size]
        test_data = ts_data.iloc[train_size:]
        
        # 모델 파라미터 설정
        order = (1, 1, 1)  # (p, d, q)
        seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s)
        
        # SARIMAX 모델 구축
        model = SARIMAX(
            train_data[target_var],
            exog=train_data[[exog_var]],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # 모델 학습
        results = model.fit(disp=False)
        
        # 모델 요약 저장
        with open(os.path.join(results_dir, f'{target_var}_sarimax_summary.txt'), 'w') as f:
            f.write(str(results.summary()))
        
        # 테스트 데이터에 대한 예측
        pred = results.get_forecast(steps=len(test_data), exog=test_data[[exog_var]])
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
        plt.title(f'SARIMAX Forecast: {target_var.replace("_", " ").title()}')
        plt.xlabel('Date')
        plt.ylabel('Search Volume')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{target_var}_sarimax_forecast.png'))
        plt.close()
        
        # 미래 예측
        future_exog = pd.DataFrame({exog_var: [ts_data[exog_var].mean()] * forecast_steps}, 
                                  index=pd.date_range(start=ts_data.index[-1], periods=forecast_steps+1, freq='D')[1:])
        
        future_pred = results.get_forecast(steps=forecast_steps, exog=future_exog)
        future_pred_ci = future_pred.conf_int()
        
        # 미래 예측 결과 시각화
        plt.figure(figsize=(12, 6))
        
        # 과거 데이터
        plt.plot(ts_data.index, ts_data[target_var], label='Historical Data')
        
        # 미래 예측
        plt.plot(future_exog.index, future_pred.predicted_mean, label='Future Forecast', color='red')
        
        # 신뢰 구간
        plt.fill_between(future_exog.index,
                        future_pred_ci.iloc[:, 0],
                        future_pred_ci.iloc[:, 1], color='pink', alpha=0.3)
        
        # 그래프 설정
        plt.title(f'SARIMAX Future Forecast: {target_var.replace("_", " ").title()}')
        plt.xlabel('Date')
        plt.ylabel('Search Volume')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{target_var}_sarimax_future_forecast.png'))
        plt.close()
        
        # 모델 성능 평가
        mse = ((pred.predicted_mean - test_data[target_var]) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(pred.predicted_mean - test_data[target_var]).mean()
        
        # 결과 저장
        with open(os.path.join(results_dir, f'{target_var}_sarimax_performance.txt'), 'w') as f:
            f.write(f"# {target_var.replace('_', ' ').title()} SARIMAX 모델 성능\n\n")
            f.write(f"- 모델 파라미터:\n")
            f.write(f"  - order: {order}\n")
            f.write(f"  - seasonal_order: {seasonal_order}\n\n")
            f.write(f"- 성능 지표:\n")
            f.write(f"  - MSE: {mse:.4f}\n")
            f.write(f"  - RMSE: {rmse:.4f}\n")
            f.write(f"  - MAE: {mae:.4f}\n")
    
    print("SARIMAX 모델 구축 완료")

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
                applied_seasonal_diff = True
                is_stationary = True
        
        # 훈련 데이터와 테스트 데이터 분리
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data.iloc[:train_size]
        test_data = ts_data.iloc[train_size:]
        
        # SARIMAX 모델 파라미터 설정
        # 차분 적용 여부에 따라 d 값 조정
        d = 1 if applied_differencing else 0
        # 계절 차분 적용 여부에 따라 D 값 조정
        D = 1 if applied_seasonal_diff else 0
        
        order = (1, d, 1)  # (p, d, q)
        seasonal_order = (1, D, 1, 12)  # (P, D, Q, s)
        
        # SARIMAX 모델 구축
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
        
        # 미래 예측
        future_exog = pd.DataFrame({f'{exog_var}_scaled': [ts_data[f'{exog_var}_scaled'].mean()] * forecast_steps}, 
                                  index=pd.date_range(start=ts_data.index[-1], periods=forecast_steps+1, freq='D')[1:])
        
        future_pred = results.get_forecast(steps=forecast_steps, exog=future_exog)
        future_pred_ci = future_pred.conf_int()
        
        # 미래 예측 결과 시각화
        plt.figure(figsize=(12, 6))
        
        # 과거 데이터
        plt.plot(ts_data.index, ts_data[target_var], label='Historical Data')
        
        # 미래 예측
        plt.plot(future_exog.index, future_pred.predicted_mean, label='Future Forecast', color='red')
        
        # 신뢰 구간
        plt.fill_between(future_exog.index,
                        future_pred_ci.iloc[:, 0],
                        future_pred_ci.iloc[:, 1], color='pink', alpha=0.3)
        
        # 그래프 설정
        plt.title(f'Improved SARIMAX Future Forecast: {target_var.replace("_", " ").title()}')
        plt.xlabel('Date')
        plt.ylabel('Search Volume')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{target_var}_improved_sarimax_future_forecast.png'))
        plt.close()
        
        # 모델 성능 평가
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
    
    print("개선된 SARIMAX 모델 구축 완료")

def statistical_modeling(data):
    """통계적 모델링 수행"""
    print("\n통계적 모델링 수행 중...")
    
    # 회귀 분석 수행
    regression_results = perform_regression_analysis(data)
    
    # 시차 효과를 고려한 회귀 분석
    lag_regression_results = analyze_lag_regression(data)
    
    # 계절성을 고려한 회귀 분석
    seasonal_regression_results = analyze_seasonal_regression(data)
    
    # 시계열 분해 분석
    perform_time_series_decomposition(data)
    
    # 시계열 정상성 검정
    stationarity_results = check_stationarity(data)
    
    # SARIMAX 모델 구축
    build_sarimax_model(data)
    
    # 개선된 SARIMAX 모델 구축
    build_improved_sarimax_model(data)
    
    print("\n통계적 모델링 완료!")

if __name__ == "__main__":
    # 데이터 로드
    data = load_processed_data()
    
    # 통계적 모델링 수행
    statistical_modeling(data)
    
    print("\n모든 분석 완료!")

