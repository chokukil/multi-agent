#!/usr/bin/env python3
"""
MCP Tool: Time Series Analysis Specialist
시계열 데이터 분석 전문 도구 - 트렌드, 계절성, 예측, 분해 분석
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import uvicorn

# FastMCP import
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Get port from environment variable
SERVER_PORT = int(os.getenv('SERVER_PORT', '8013'))

# FastMCP 서버 생성
mcp = FastMCP("Time Series Analysis Specialist")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """시계열 분석 전문 클래스"""
    
    @staticmethod
    def analyze_trend(data: List[float], dates: List[str] = None) -> Dict[str, Any]:
        """트렌드 분석"""
        try:
            from scipy import stats
            
            if not data or len(data) < 3:
                return {"error": "트렌드 분석을 위해서는 최소 3개 이상의 데이터 포인트가 필요합니다."}
            
            # 선형 회귀를 통한 트렌드 분석
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            # 트렌드 방향 및 강도 판정
            trend_direction = "증가" if slope > 0 else "감소" if slope < 0 else "평평"
            trend_strength = abs(r_value)
            
            if trend_strength > 0.8:
                strength_desc = "매우 강함"
            elif trend_strength > 0.6:
                strength_desc = "강함"
            elif trend_strength > 0.4:
                strength_desc = "보통"
            elif trend_strength > 0.2:
                strength_desc = "약함"
            else:
                strength_desc = "매우 약함"
            
            # 이동평균 계산
            ma_7 = pd.Series(data).rolling(window=min(7, len(data)//2)).mean().tolist()
            ma_30 = pd.Series(data).rolling(window=min(30, len(data)//2)).mean().tolist()
            
            return {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "strength_description": strength_desc,
                "moving_average_7": ma_7,
                "moving_average_30": ma_30,
                "trend_line": [slope * i + intercept for i in x],
                "interpretation": f"데이터는 {trend_direction} 트렌드를 보이며, 트렌드 강도는 {strength_desc}입니다. (R² = {r_value**2:.3f})"
            }
            
        except Exception as e:
            return {"error": f"트렌드 분석 중 오류가 발생했습니다: {str(e)}"}
    
    @staticmethod
    def detect_seasonality(data: List[float], period: int = None) -> Dict[str, Any]:
        """계절성 탐지"""
        try:
            if len(data) < 14:
                return {"error": "계절성 분석을 위해서는 최소 14개 이상의 데이터 포인트가 필요합니다."}
            
            series = pd.Series(data)
            
            # 자동 주기 탐지 (제공되지 않은 경우)
            if period is None:
                # 일반적인 주기들 테스트
                common_periods = [7, 14, 30, 365]
                best_period = None
                best_correlation = 0
                
                for p in common_periods:
                    if len(data) >= 2 * p:
                        # 지연 자기상관 계산
                        correlation = series.autocorr(lag=p)
                        if correlation > best_correlation:
                            best_correlation = correlation
                            best_period = p
                
                period = best_period if best_correlation > 0.3 else None
            
            if period and len(data) >= 2 * period:
                # 계절성 분해
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                decomposition = seasonal_decompose(series, model='additive', period=period)
                
                # 계절성 강도 계산
                seasonal_strength = np.var(decomposition.seasonal) / np.var(series)
                
                return {
                    "has_seasonality": True,
                    "period": period,
                    "seasonal_strength": seasonal_strength,
                    "seasonal_component": decomposition.seasonal.tolist(),
                    "trend_component": decomposition.trend.dropna().tolist(),
                    "residual_component": decomposition.resid.dropna().tolist(),
                    "interpretation": f"주기 {period}의 계절성 패턴이 감지되었습니다. 계절성 강도: {seasonal_strength:.3f}"
                }
            else:
                return {
                    "has_seasonality": False,
                    "period": None,
                    "interpretation": "명확한 계절성 패턴을 감지할 수 없습니다."
                }
                
        except Exception as e:
            return {"error": f"계절성 분석 중 오류가 발생했습니다: {str(e)}"}
    
    @staticmethod
    def test_stationarity(data: List[float]) -> Dict[str, Any]:
        """정상성 검정"""
        try:
            if len(data) < 10:
                return {"error": "정상성 검정을 위해서는 최소 10개 이상의 데이터 포인트가 필요합니다."}
            
            from statsmodels.tsa.stattools import adfuller, kpss
            
            series = pd.Series(data).dropna()
            
            # ADF 검정 (단위근 검정)
            adf_result = adfuller(series)
            adf_stationary = adf_result[1] < 0.05
            
            # KPSS 검정 (추세 정상성 검정)
            try:
                kpss_result = kpss(series, regression='ct')
                kpss_stationary = kpss_result[1] > 0.05
            except:
                kpss_result = None
                kpss_stationary = None
            
            # 종합 판정
            if adf_stationary and (kpss_stationary or kpss_stationary is None):
                conclusion = "정상성"
            elif not adf_stationary and (not kpss_stationary if kpss_stationary is not None else True):
                conclusion = "비정상성"
            else:
                conclusion = "불확실"
            
            result = {
                "adf_test": {
                    "statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "critical_values": adf_result[4],
                    "is_stationary": adf_stationary
                },
                "conclusion": conclusion,
                "interpretation": f"ADF 검정 결과: {'정상성' if adf_stationary else '비정상성'}"
            }
            
            if kpss_result:
                result["kpss_test"] = {
                    "statistic": kpss_result[0],
                    "p_value": kpss_result[1],
                    "critical_values": kpss_result[3],
                    "is_stationary": kpss_stationary
                }
                result["interpretation"] += f", KPSS 검정 결과: {'정상성' if kpss_stationary else '비정상성'}"
            
            return result
            
        except Exception as e:
            return {"error": f"정상성 검정 중 오류가 발생했습니다: {str(e)}"}
    
    @staticmethod
    def forecast_timeseries(data: List[float], periods: int = 10, method: str = "auto") -> Dict[str, Any]:
        """시계열 예측"""
        try:
            if len(data) < 5:
                return {"error": "예측을 위해서는 최소 5개 이상의 데이터 포인트가 필요합니다."}
            
            series = pd.Series(data)
            
            # 예측 방법 선택
            if method == "auto":
                # 데이터 특성에 따른 자동 선택
                if len(data) >= 24:
                    method = "exponential_smoothing"
                else:
                    method = "linear_trend"
            
            if method == "linear_trend":
                # 선형 트렌드 기반 예측
                from scipy import stats
                x = np.arange(len(data))
                slope, intercept, _, _, _ = stats.linregress(x, data)
                
                forecast_x = np.arange(len(data), len(data) + periods)
                forecast = [slope * i + intercept for i in forecast_x]
                
                return {
                    "method": "linear_trend",
                    "periods": periods,
                    "forecast": forecast,
                    "interpretation": f"선형 트렌드 모델로 {periods}기간 예측을 수행했습니다."
                }
                
            elif method == "exponential_smoothing":
                # 지수평활법
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
                # 단순 지수평활
                model = ExponentialSmoothing(series, trend='add')
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=periods).tolist()
                
                return {
                    "method": "exponential_smoothing",
                    "periods": periods,
                    "forecast": forecast,
                    "model_aic": fitted_model.aic,
                    "interpretation": f"지수평활법으로 {periods}기간 예측을 수행했습니다. AIC: {fitted_model.aic:.2f}"
                }
                
            elif method == "arima":
                # ARIMA 모델
                from statsmodels.tsa.arima.model import ARIMA
                
                # 간단한 ARIMA(1,1,1) 모델
                model = ARIMA(series, order=(1, 1, 1))
                fitted_model = model.fit()
                forecast_result = fitted_model.forecast(steps=periods)
                confidence_interval = fitted_model.get_forecast(steps=periods).conf_int()
                
                return {
                    "method": "arima",
                    "periods": periods,
                    "forecast": forecast_result.tolist(),
                    "confidence_interval": {
                        "lower": confidence_interval.iloc[:, 0].tolist(),
                        "upper": confidence_interval.iloc[:, 1].tolist()
                    },
                    "model_aic": fitted_model.aic,
                    "interpretation": f"ARIMA(1,1,1) 모델로 {periods}기간 예측을 수행했습니다. AIC: {fitted_model.aic:.2f}"
                }
                
        except Exception as e:
            return {"error": f"예측 중 오류가 발생했습니다: {str(e)}"}
    
    @staticmethod
    def detect_anomalies(data: List[float], method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
        """이상치 탐지"""
        try:
            if len(data) < 5:
                return {"error": "이상치 탐지를 위해서는 최소 5개 이상의 데이터 포인트가 필요합니다."}
            
            series = pd.Series(data)
            anomalies = []
            anomaly_indices = []
            
            if method == "zscore":
                # Z-score 기반 이상치 탐지
                z_scores = np.abs(stats.zscore(series))
                anomaly_mask = z_scores > threshold
                anomalies = series[anomaly_mask].tolist()
                anomaly_indices = series[anomaly_mask].index.tolist()
                
            elif method == "iqr":
                # IQR 기반 이상치 탐지
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                anomaly_mask = (series < lower_bound) | (series > upper_bound)
                anomalies = series[anomaly_mask].tolist()
                anomaly_indices = series[anomaly_mask].index.tolist()
                
            elif method == "isolation_forest":
                # Isolation Forest (scikit-learn 필요)
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    
                    anomaly_mask = outlier_labels == -1
                    anomalies = series[anomaly_mask].tolist()
                    anomaly_indices = series[anomaly_mask].index.tolist()
                    
                except ImportError:
                    return {"error": "Isolation Forest 방법을 사용하려면 scikit-learn이 필요합니다."}
            
            return {
                "method": method,
                "threshold": threshold,
                "anomaly_count": len(anomalies),
                "anomaly_indices": anomaly_indices,
                "anomaly_values": anomalies,
                "anomaly_percentage": (len(anomalies) / len(data)) * 100,
                "interpretation": f"{method} 방법으로 {len(anomalies)}개의 이상치를 탐지했습니다. ({(len(anomalies)/len(data)*100):.1f}%)"
            }
            
        except Exception as e:
            return {"error": f"이상치 탐지 중 오류가 발생했습니다: {str(e)}"}
    
    @staticmethod
    def calculate_autocorrelation(data: List[float], max_lags: int = 20) -> Dict[str, Any]:
        """자기상관 계산"""
        try:
            if len(data) < max_lags + 1:
                max_lags = len(data) - 1
            
            series = pd.Series(data)
            autocorrelations = []
            
            for lag in range(1, max_lags + 1):
                autocorr = series.autocorr(lag=lag)
                autocorrelations.append(autocorr if not pd.isna(autocorr) else 0.0)
            
            # 유의한 자기상관 찾기
            significant_lags = []
            threshold = 1.96 / np.sqrt(len(data))  # 95% 신뢰구간
            
            for i, autocorr in enumerate(autocorrelations):
                if abs(autocorr) > threshold:
                    significant_lags.append({
                        "lag": i + 1,
                        "autocorrelation": autocorr
                    })
            
            return {
                "max_lags": max_lags,
                "autocorrelations": autocorrelations,
                "significant_lags": significant_lags,
                "significance_threshold": threshold,
                "interpretation": f"최대 {max_lags}까지의 자기상관을 계산했습니다. {len(significant_lags)}개의 유의한 지연이 발견되었습니다."
            }
            
        except Exception as e:
            return {"error": f"자기상관 계산 중 오류가 발생했습니다: {str(e)}"}


# MCP 도구 등록
@mcp.tool
def analyze_trend(data: List[float], dates: List[str] = None) -> Dict[str, Any]:
    """
    시계열 데이터의 트렌드를 분석합니다.
    
    Args:
        data: 분석할 시계열 데이터 (숫자 리스트)
        dates: 날짜 정보 (선택사항)
    
    Returns:
        트렌드 분석 결과 (방향, 강도, 통계량 등)
    """
    return TimeSeriesAnalyzer.analyze_trend(data, dates)


@mcp.tool  
def detect_seasonality(data: List[float], period: int = None) -> Dict[str, Any]:
    """
    시계열 데이터의 계절성을 탐지하고 분석합니다.
    
    Args:
        data: 분석할 시계열 데이터
        period: 계절성 주기 (자동 탐지시 None)
    
    Returns:
        계절성 분석 결과
    """
    return TimeSeriesAnalyzer.detect_seasonality(data, period)


@mcp.tool
def test_stationarity(data: List[float]) -> Dict[str, Any]:
    """
    시계열 데이터의 정상성을 검정합니다.
    
    Args:
        data: 검정할 시계열 데이터
    
    Returns:
        정상성 검정 결과 (ADF, KPSS 검정)
    """
    return TimeSeriesAnalyzer.test_stationarity(data)


@mcp.tool
def forecast_timeseries(data: List[float], periods: int = 10, method: str = "auto") -> Dict[str, Any]:
    """
    시계열 데이터를 예측합니다.
    
    Args:
        data: 예측할 시계열 데이터
        periods: 예측 기간
        method: 예측 방법 (auto, linear_trend, exponential_smoothing, arima)
    
    Returns:
        예측 결과
    """
    return TimeSeriesAnalyzer.forecast_timeseries(data, periods, method)


@mcp.tool
def detect_anomalies(data: List[float], method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
    """
    시계열 데이터의 이상치를 탐지합니다.
    
    Args:
        data: 분석할 시계열 데이터
        method: 탐지 방법 (zscore, iqr, isolation_forest)
        threshold: 이상치 임계값
    
    Returns:
        이상치 탐지 결과
    """
    return TimeSeriesAnalyzer.detect_anomalies(data, method, threshold)


@mcp.tool
def calculate_autocorrelation(data: List[float], max_lags: int = 20) -> Dict[str, Any]:
    """
    시계열 데이터의 자기상관을 계산합니다.
    
    Args:
        data: 분석할 시계열 데이터
        max_lags: 최대 지연 수
    
    Returns:
        자기상관 분석 결과
    """
    return TimeSeriesAnalyzer.calculate_autocorrelation(data, max_lags)


@mcp.tool
def decompose_timeseries(data: List[float], model: str = "additive", period: int = None) -> Dict[str, Any]:
    """
    시계열을 트렌드, 계절성, 잔차로 분해합니다.
    
    Args:
        data: 분해할 시계열 데이터
        model: 분해 모델 (additive, multiplicative)
        period: 계절성 주기
    
    Returns:
        분해 결과
    """
    try:
        if len(data) < 14:
            return {"error": "시계열 분해를 위해서는 최소 14개 이상의 데이터 포인트가 필요합니다."}
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        series = pd.Series(data)
        
        # 주기 자동 설정
        if period is None:
            period = min(12, len(data) // 2)
        
        decomposition = seasonal_decompose(series, model=model, period=period)
        
        return {
            "model": model,
            "period": period,
            "trend": decomposition.trend.dropna().tolist(),
            "seasonal": decomposition.seasonal.tolist(),
            "residual": decomposition.resid.dropna().tolist(),
            "original": data,
            "interpretation": f"{model} 모델로 주기 {period}의 시계열 분해를 수행했습니다."
        }
        
    except Exception as e:
        return {"error": f"시계열 분해 중 오류가 발생했습니다: {str(e)}"}


@mcp.tool
def comprehensive_timeseries_analysis(data: List[float], 
                                    dates: List[str] = None,
                                    forecast_periods: int = 10,
                                    seasonal_period: int = None) -> Dict[str, Any]:
    """
    종합적인 시계열 분석을 수행합니다.
    
    Args:
        data: 분석할 시계열 데이터
        dates: 날짜 정보
        forecast_periods: 예측 기간
        seasonal_period: 계절성 주기
    
    Returns:
        종합 분석 결과
    """
    try:
        if len(data) < 10:
            return {"error": "종합 분석을 위해서는 최소 10개 이상의 데이터 포인트가 필요합니다."}
        
        results = {
            "data_summary": {
                "length": len(data),
                "mean": np.mean(data),
                "std": np.std(data),
                "min": np.min(data),
                "max": np.max(data)
            }
        }
        
        # 각 분석 수행
        results["trend_analysis"] = TimeSeriesAnalyzer.analyze_trend(data, dates)
        results["seasonality_analysis"] = TimeSeriesAnalyzer.detect_seasonality(data, seasonal_period)
        results["stationarity_test"] = TimeSeriesAnalyzer.test_stationarity(data)
        results["anomaly_detection"] = TimeSeriesAnalyzer.detect_anomalies(data)
        results["autocorrelation"] = TimeSeriesAnalyzer.calculate_autocorrelation(data)
        results["forecast"] = TimeSeriesAnalyzer.forecast_timeseries(data, forecast_periods)
        
        # 종합 해석
        interpretation = []
        
        # 트렌드 해석
        if "error" not in results["trend_analysis"]:
            trend = results["trend_analysis"]
            interpretation.append(f"트렌드: {trend['trend_direction']} ({trend['strength_description']})")
        
        # 계절성 해석
        if "error" not in results["seasonality_analysis"]:
            seasonality = results["seasonality_analysis"]
            if seasonality["has_seasonality"]:
                interpretation.append(f"계절성: 주기 {seasonality['period']} 감지됨")
            else:
                interpretation.append("계절성: 감지되지 않음")
        
        # 정상성 해석
        if "error" not in results["stationarity_test"]:
            stationarity = results["stationarity_test"]
            interpretation.append(f"정상성: {stationarity['conclusion']}")
        
        # 이상치 해석
        if "error" not in results["anomaly_detection"]:
            anomalies = results["anomaly_detection"]
            interpretation.append(f"이상치: {anomalies['anomaly_count']}개 ({anomalies['anomaly_percentage']:.1f}%)")
        
        results["interpretation"] = ". ".join(interpretation)
        results["analysis_timestamp"] = datetime.now().isoformat()
        
        return results
        
    except Exception as e:
        return {"error": f"종합 분석 중 오류가 발생했습니다: {str(e)}"}


if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Time Series Analysis MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)