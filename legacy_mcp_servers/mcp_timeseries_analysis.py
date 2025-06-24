# -*- coding: utf-8 -*-
"""
Time Series Analysis Specialist Tools - Trend, Seasonality, Forecasting, Decomposition Analysis
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
import warnings
warnings.filterwarnings('ignore')

# FastMCP import
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Get port from environment variable
SERVER_PORT = int(os.getenv('SERVER_PORT', '8013'))

# FastMCP server creation
mcp = FastMCP(
    "TimeSeriesAnalyst",
    instructions="""You are a time series analysis specialist.
    Your tools focus on:
    1. Trend analysis (direction, strength measurement)
    2. Seasonality detection and decomposition
    3. Time series forecasting (ARIMA, exponential smoothing)
    4. Statistical tests for stationarity
    5. Change point detection
    """
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """Time Series Analysis Specialist Tools"""
    
    @staticmethod
    def analyze_trend(data: List[float], time_column: Optional[List[str]] = None) -> Dict[str, Any]:
        """Trend Analysis"""
        try:
            if len(data) < 3:
                return {"error": "Trend analysis requires at least 3 data points"}
            
            # Linear regression for trend analysis
            X = np.arange(len(data)).reshape(-1, 1)
            y = np.array(data)
            
            # Trend direction and strength determination
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(data)), data)
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            
            # Trend strength assessment
            r_squared = r_value ** 2
            if r_squared > 0.8:
                strength_desc = "very strong"
            elif r_squared > 0.6:
                strength_desc = "strong"
            elif r_squared > 0.4:
                strength_desc = "moderate"
            elif r_squared > 0.2:
                strength_desc = "weak"
            else:
                strength_desc = "very weak"
            
            # Moving average calculation
            window_size = min(5, len(data) // 3)
            if window_size >= 2:
                moving_avg = pd.Series(data).rolling(window=window_size).mean().tolist()
            else:
                moving_avg = data
            
            return {
                "trend_direction": trend_direction,
                "slope": slope,
                "r_squared": r_squared,
                "p_value": p_value,
                "strength": strength_desc,
                "moving_average": moving_avg,
                "trend_line": [intercept + slope * i for i in range(len(data))],
                "interpretation": f"The data shows a {trend_direction} trend with {strength_desc} strength (R² = {r_value**2:.3f})"
            }
        except Exception as e:
            return {"error": f"Error in trend analysis: {str(e)}"}
    
    @staticmethod
    def detect_seasonality(data: List[float], period: Optional[int] = None) -> Dict[str, Any]:
        """Seasonality Detection"""
        try:
            if len(data) < 14:
                return {"error": "Seasonality analysis requires at least 14 data points"}
            
            series = pd.Series(data)
            
            # Automatic period detection (common periods)
            if period is None:
                # Common periods: 7, 14, 30, 365
                common_periods = [7, 14, 30, 365]
                best_period = None
                best_correlation = 0
                
                for p in common_periods:
                    if len(data) >= 2 * p:
                        # Calculate autocorrelation
                        correlation = series.autocorr(lag=p)
                        if correlation > best_correlation:
                            best_correlation = correlation
                            best_period = p
                
                period = best_period if best_correlation > 0.3 else None
            
            if period and len(data) >= 2 * period:
                # Seasonality decomposition
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                decomposition = seasonal_decompose(series, model='additive', period=period)
                
                # Seasonal strength calculation
                seasonal_strength = np.var(decomposition.seasonal) / np.var(series)
                
                return {
                    "has_seasonality": True,
                    "period": period,
                    "seasonal_strength": seasonal_strength,
                    "seasonal_component": decomposition.seasonal.tolist(),
                    "trend_component": decomposition.trend.dropna().tolist(),
                    "residual_component": decomposition.resid.dropna().tolist(),
                    "interpretation": f"Seasonal period {period} detected with seasonal strength: {seasonal_strength:.3f}"
                }
            else:
                return {
                    "has_seasonality": False,
                    "period": None,
                    "interpretation": "No clear seasonality detected."
                }
                
        except Exception as e:
            return {"error": f"Error in seasonality detection: {str(e)}"}
    
    @staticmethod
    def test_stationarity(data: List[float]) -> Dict[str, Any]:
        """Stationarity Test"""
        try:
            if len(data) < 10:
                return {"error": "Stationarity test requires at least 10 data points"}
            
            from statsmodels.tsa.stattools import adfuller, kpss
            
            series = pd.Series(data).dropna()
            
            # ADF test (Augmented Dickey-Fuller test)
            adf_result = adfuller(series)
            adf_stationary = adf_result[1] < 0.05
            
            # KPSS test (Kwiatkowski-Phillips-Schmidt-Shin test)
            try:
                kpss_result = kpss(series, regression='ct')
                kpss_stationary = kpss_result[1] > 0.05
            except:
                kpss_result = None
                kpss_stationary = None
            
            # Comprehensive conclusion
            if adf_stationary and (kpss_stationary or kpss_stationary is None):
                conclusion = "Stationary"
            elif not adf_stationary and (not kpss_stationary if kpss_stationary is not None else True):
                conclusion = "Non-stationary"
            else:
                conclusion = "Uncertain"
            
            result = {
                "adf_test": {
                    "statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "critical_values": adf_result[4],
                    "is_stationary": adf_stationary
                },
                "conclusion": conclusion,
                "interpretation": f"ADF test conclusion: {'Stationary' if adf_stationary else 'Non-stationary'}"
            }
            
            if kpss_result:
                result["kpss_test"] = {
                    "statistic": kpss_result[0],
                    "p_value": kpss_result[1],
                    "critical_values": kpss_result[3],
                    "is_stationary": kpss_stationary
                }
                result["interpretation"] += f", KPSS test conclusion: {'Stationary' if kpss_stationary else 'Non-stationary'}"
            
            return result
            
        except Exception as e:
            return {"error": f"Error in stationarity test: {str(e)}"}
    
    @staticmethod
    def forecast_timeseries(data: List[float], periods: int = 10, method: str = "auto") -> Dict[str, Any]:
        """Time Series Forecasting"""
        try:
            if len(data) < 5:
                return {"error": "Forecasting requires at least 5 data points"}
            
            series = pd.Series(data)
            
            # Automatic method selection
            if method == "auto":
                # Automatic selection based on data length
                if len(data) >= 24:
                    method = "exponential_smoothing"
                else:
                    method = "linear_trend"
            
            if method == "linear_trend":
                # Linear regression-based forecasting
                from scipy import stats
                x = np.arange(len(data))
                slope, intercept, _, _, _ = stats.linregress(x, data)
                
                forecast_x = np.arange(len(data), len(data) + periods)
                forecast = [slope * i + intercept for i in forecast_x]
                
                return {
                    "method": "linear_trend",
                    "periods": periods,
                    "forecast": forecast,
                    "interpretation": f"Linear trend model forecast for {periods} periods."
                }
                
            elif method == "exponential_smoothing":
                # Exponential smoothing
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
                # Fit exponential smoothing model
                model = ExponentialSmoothing(series, trend='add')
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=periods).tolist()
                
                return {
                    "method": "exponential_smoothing",
                    "periods": periods,
                    "forecast": forecast,
                    "model_aic": fitted_model.aic,
                    "interpretation": f"Exponential smoothing forecast for {periods} periods. AIC: {fitted_model.aic:.2f}"
                }
                
            elif method == "arima":
                # ARIMA model
                from statsmodels.tsa.arima.model import ARIMA
                
                # Simple ARIMA(1,1,1) model
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
                    "interpretation": f"ARIMA(1,1,1) model forecast for {periods} periods. AIC: {fitted_model.aic:.2f}"
                }
                
        except Exception as e:
            return {"error": f"Error in forecasting: {str(e)}"}
    
    @staticmethod
    def detect_anomalies(data: List[float], method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
        """Anomaly Detection"""
        try:
            if len(data) < 5:
                return {"error": "Anomaly detection requires at least 5 data points"}
            
            series = pd.Series(data)
            anomalies = []
            anomaly_indices = []
            
            if method == "zscore":
                # Z-score based anomaly detection
                z_scores = np.abs(stats.zscore(series))
                anomaly_mask = z_scores > threshold
                anomalies = series[anomaly_mask].tolist()
                anomaly_indices = series[anomaly_mask].index.tolist()
                
            elif method == "iqr":
                # IQR based anomaly detection
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                anomaly_mask = (series < lower_bound) | (series > upper_bound)
                anomalies = series[anomaly_mask].tolist()
                anomaly_indices = series[anomaly_mask].index.tolist()
                
            elif method == "isolation_forest":
                # Isolation Forest (scikit-learn required)
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    
                    anomaly_mask = outlier_labels == -1
                    anomalies = series[anomaly_mask].tolist()
                    anomaly_indices = series[anomaly_mask].index.tolist()
                    
                except ImportError:
                    return {"error": "Isolation Forest method requires scikit-learn"}
            
            return {
                "method": method,
                "threshold": threshold,
                "anomaly_count": len(anomalies),
                "anomaly_indices": anomaly_indices,
                "anomaly_values": anomalies,
                "anomaly_percentage": (len(anomalies) / len(data)) * 100,
                "interpretation": f"{method} method detected {len(anomalies)} anomalies. ({len(anomalies)/len(data)*100:.1f}%)"
            }
            
        except Exception as e:
            return {"error": f"Error in anomaly detection: {str(e)}"}
    
    @staticmethod
    def calculate_autocorrelation(data: List[float], max_lags: int = 20) -> Dict[str, Any]:
        """Autocorrelation Calculation"""
        try:
            if len(data) < max_lags + 1:
                max_lags = len(data) - 1
            
            series = pd.Series(data)
            autocorrelations = []
            
            for lag in range(1, max_lags + 1):
                autocorr = series.autocorr(lag=lag)
                autocorrelations.append(autocorr if not pd.isna(autocorr) else 0.0)
            
            # Find significant lags
            significant_lags = []
            threshold = 1.96 / np.sqrt(len(data))  # 95% confidence interval
            
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
                "interpretation": f"Autocorrelation calculation completed. {len(significant_lags)} significant lags found."
            }
            
        except Exception as e:
            return {"error": f"Error in autocorrelation calculation: {str(e)}"}


# MCP server implementation
@mcp.tool("analyze_trend")
def analyze_trend(data: List[float], time_column: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Trend Analysis
    
    Args:
        data: List of data points
        time_column: Optional list of time stamps
    
    Returns:
        Trend analysis result (direction, strength, interpretation)
    """
    return TimeSeriesAnalyzer.analyze_trend(data, time_column)


@mcp.tool("detect_seasonality")
def detect_seasonality(data: List[float], period: Optional[int] = None) -> Dict[str, Any]:
    """
    Seasonality Detection
    
    Args:
        data: List of data points
        period: Optional seasonal period
    
    Returns:
        Seasonality analysis result
    """
    return TimeSeriesAnalyzer.detect_seasonality(data, period)


@mcp.tool("test_stationarity")
def test_stationarity(data: List[float]) -> Dict[str, Any]:
    """
    Stationarity Test
    
    Args:
        data: List of data points
    
    Returns:
        Stationarity test result (ADF, KPSS test results)
    """
    return TimeSeriesAnalyzer.test_stationarity(data)


@mcp.tool("forecast_timeseries")
def forecast_timeseries(data: List[float], periods: int = 10, method: str = "auto") -> Dict[str, Any]:
    """
    Time Series Forecasting
    
    Args:
        data: List of data points
        periods: Number of forecast periods
        method: Forecasting method (auto, linear_trend, exponential_smoothing, arima)
    
    Returns:
        Forecasting result
    """
    return TimeSeriesAnalyzer.forecast_timeseries(data, periods, method)


@mcp.tool("detect_anomalies")
def detect_anomalies(data: List[float], method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
    """
    Anomaly Detection
    
    Args:
        data: List of data points
        method: Anomaly detection method (zscore, iqr, isolation_forest)
        threshold: Anomaly threshold
    
    Returns:
        Anomaly detection result
    """
    return TimeSeriesAnalyzer.detect_anomalies(data, method, threshold)


@mcp.tool("calculate_autocorrelation")
def calculate_autocorrelation(data: List[float], max_lags: int = 20) -> Dict[str, Any]:
    """
    Autocorrelation Calculation
    
    Args:
        data: List of data points
        max_lags: Maximum number of lags
    
    Returns:
        Autocorrelation analysis result
    """
    return TimeSeriesAnalyzer.calculate_autocorrelation(data, max_lags)


@mcp.tool("decompose_timeseries")
def decompose_timeseries(data: List[float], model: str = "additive", period: int = None) -> Dict[str, Any]:
    """
    Time Series Decomposition
    
    Args:
        data: List of data points to decompose
        model: Decomposition model (additive, multiplicative)
        period: Seasonal period
    
    Returns:
        Decomposition result
    """
    try:
        if len(data) < 14:
            return {"error": "Decomposition requires at least 14 data points"}
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        series = pd.Series(data)
        
        # Automatic period detection
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
            "interpretation": f"{model} model performed decomposition for period {period}."
        }
        
    except Exception as e:
        return {"error": f"Error in decomposition: {str(e)}"}


@mcp.tool("comprehensive_timeseries_analysis")
def comprehensive_timeseries_analysis(data: List[float], 
                                    dates: List[str] = None,
                                    forecast_periods: int = 10,
                                    seasonal_period: int = None) -> Dict[str, Any]:
    """
    Comprehensive Time Series Analysis
    
    Args:
        data: List of data points
        dates: Optional list of dates
        forecast_periods: Number of forecast periods
        seasonal_period: Optional seasonal period
    
    Returns:
        Comprehensive analysis result
    """
    try:
        if len(data) < 10:
            return {"error": "Comprehensive analysis requires at least 10 data points"}
        
        results = {
            "data_summary": {
                "length": len(data),
                "mean": np.mean(data),
                "std": np.std(data),
                "min": np.min(data),
                "max": np.max(data)
            }
        }
        
        # Individual analysis
        results["trend_analysis"] = TimeSeriesAnalyzer.analyze_trend(data, dates)
        results["seasonality_analysis"] = TimeSeriesAnalyzer.detect_seasonality(data, seasonal_period)
        results["stationarity_test"] = TimeSeriesAnalyzer.test_stationarity(data)
        results["anomaly_detection"] = TimeSeriesAnalyzer.detect_anomalies(data)
        results["autocorrelation"] = TimeSeriesAnalyzer.calculate_autocorrelation(data)
        results["forecast"] = TimeSeriesAnalyzer.forecast_timeseries(data, forecast_periods)
        
        # Comprehensive interpretation
        interpretation = []
        
        # Trend analysis
        if "error" not in results["trend_analysis"]:
            trend = results["trend_analysis"]
            interpretation.append(f"Trend: {trend['trend_direction']} ({trend['strength']})")
        
        # Seasonality analysis
        if "error" not in results["seasonality_analysis"]:
            seasonality = results["seasonality_analysis"]
            if seasonality["has_seasonality"]:
                interpretation.append(f"Seasonal period: {seasonality['period']}")
            else:
                interpretation.append("No clear seasonality detected")
        
        # Stationarity test
        if "error" not in results["stationarity_test"]:
            stationarity = results["stationarity_test"]
            interpretation.append(f"Stationarity: {stationarity['conclusion']}")
        
        # Anomaly detection
        if "error" not in results["anomaly_detection"]:
            anomalies = results["anomaly_detection"]
            interpretation.append(f"Anomalies: {anomalies['anomaly_count']} ({anomalies['anomaly_percentage']:.1f}%)")
        
        results["interpretation"] = ". ".join(interpretation)
        results["analysis_timestamp"] = datetime.now().isoformat()
        
        return results
        
    except Exception as e:
        return {"error": f"Error in comprehensive analysis: {str(e)}"}


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
