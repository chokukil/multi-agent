#!/usr/bin/env python3
"""
MCP Tool: Anomaly Detection Specialist
이상 탐지 전문 도구 - 통계적, 기계학습, 앙상블 기반 이상치 탐지
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import uvicorn

# FastMCP import
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Get port from environment variable
SERVER_PORT = int(os.getenv('SERVER_PORT', '8014'))

# FastMCP 서버 생성
mcp = FastMCP("Anomaly Detection Specialist")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """이상 탐지 전문 클래스"""
    
    @staticmethod
    def statistical_outliers(data: List[float], method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
        """통계적 이상치 탐지"""
        try:
            if not data or len(data) < 3:
                return {"error": "최소 3개 이상의 데이터 포인트가 필요합니다."}
            
            series = pd.Series(data)
            outliers = []
            outlier_indices = []
            
            if method == "zscore":
                # Z-score 방법
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > threshold
                outliers = series[outlier_mask].tolist()
                outlier_indices = series[outlier_mask].index.tolist()
                scores = z_scores.tolist()
                
            elif method == "iqr":
                # IQR 방법
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                outliers = series[outlier_mask].tolist()
                outlier_indices = series[outlier_mask].index.tolist()
                scores = ((series - series.median()) / IQR).abs().tolist()
                
            elif method == "modified_zscore":
                # Modified Z-score (중앙값 기반)
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
                outliers = series[outlier_mask].tolist()
                outlier_indices = series[outlier_mask].index.tolist()
                scores = np.abs(modified_z_scores).tolist()
                
            else:
                return {"error": f"지원되지 않는 방법: {method}"}
            
            return {
                "method": method,
                "threshold": threshold,
                "outliers": outliers,
                "outlier_indices": outlier_indices,
                "outlier_count": len(outliers),
                "outlier_percentage": (len(outliers) / len(data)) * 100,
                "scores": scores,
                "statistics": {
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max()
                },
                "interpretation": f"{method} 방법으로 {len(outliers)}개의 이상치를 탐지했습니다 ({(len(outliers)/len(data)*100):.1f}%)"
            }
            
        except Exception as e:
            return {"error": f"통계적 이상치 탐지 중 오류: {str(e)}"}
    
    @staticmethod
    def isolation_forest_detection(data: List[List[float]], contamination: float = 0.1) -> Dict[str, Any]:
        """Isolation Forest 이상 탐지"""
        try:
            from sklearn.ensemble import IsolationForest
            
            if not data or len(data) < 5:
                return {"error": "최소 5개 이상의 데이터 포인트가 필요합니다."}
            
            # 데이터 변환
            if isinstance(data[0], (int, float)):
                # 1차원 데이터인 경우
                X = np.array(data).reshape(-1, 1)
            else:
                # 다차원 데이터인 경우
                X = np.array(data)
            
            # 모델 학습 및 예측
            model = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = model.fit_predict(X)
            anomaly_scores = model.decision_function(X)
            
            # 결과 정리
            outliers = outlier_labels == -1
            outlier_indices = np.where(outliers)[0].tolist()
            outlier_data = X[outliers].tolist()
            
            return {
                "method": "isolation_forest",
                "contamination": contamination,
                "outlier_indices": outlier_indices,
                "outlier_data": outlier_data,
                "outlier_count": np.sum(outliers),
                "outlier_percentage": (np.sum(outliers) / len(data)) * 100,
                "anomaly_scores": anomaly_scores.tolist(),
                "interpretation": f"Isolation Forest로 {np.sum(outliers)}개의 이상치를 탐지했습니다 ({(np.sum(outliers)/len(data)*100):.1f}%)"
            }
            
        except ImportError:
            return {"error": "scikit-learn 라이브러리가 필요합니다."}
        except Exception as e:
            return {"error": f"Isolation Forest 탐지 중 오류: {str(e)}"}
    
    @staticmethod
    def local_outlier_factor(data: List[List[float]], contamination: float = 0.1, n_neighbors: int = 20) -> Dict[str, Any]:
        """Local Outlier Factor 이상 탐지"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            if not data or len(data) < 5:
                return {"error": "최소 5개 이상의 데이터 포인트가 필요합니다."}
            
            # 데이터 변환
            if isinstance(data[0], (int, float)):
                X = np.array(data).reshape(-1, 1)
            else:
                X = np.array(data)
            
            # 이웃 수 조정
            n_neighbors = min(n_neighbors, len(data) - 1)
            
            # 모델 학습 및 예측
            model = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
            outlier_labels = model.fit_predict(X)
            lof_scores = -model.negative_outlier_factor_
            
            # 결과 정리
            outliers = outlier_labels == -1
            outlier_indices = np.where(outliers)[0].tolist()
            outlier_data = X[outliers].tolist()
            
            return {
                "method": "local_outlier_factor",
                "contamination": contamination,
                "n_neighbors": n_neighbors,
                "outlier_indices": outlier_indices,
                "outlier_data": outlier_data,
                "outlier_count": np.sum(outliers),
                "outlier_percentage": (np.sum(outliers) / len(data)) * 100,
                "lof_scores": lof_scores.tolist(),
                "interpretation": f"LOF로 {np.sum(outliers)}개의 이상치를 탐지했습니다 ({(np.sum(outliers)/len(data)*100):.1f}%)"
            }
            
        except ImportError:
            return {"error": "scikit-learn 라이브러리가 필요합니다."}
        except Exception as e:
            return {"error": f"LOF 탐지 중 오류: {str(e)}"}
    
    @staticmethod
    def one_class_svm(data: List[List[float]], contamination: float = 0.1) -> Dict[str, Any]:
        """One-Class SVM 이상 탐지"""
        try:
            from sklearn.svm import OneClassSVM
            from sklearn.preprocessing import StandardScaler
            
            if not data or len(data) < 5:
                return {"error": "최소 5개 이상의 데이터 포인트가 필요합니다."}
            
            # 데이터 변환
            if isinstance(data[0], (int, float)):
                X = np.array(data).reshape(-1, 1)
            else:
                X = np.array(data)
            
            # 데이터 정규화
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 모델 학습 및 예측
            model = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
            outlier_labels = model.fit_predict(X_scaled)
            decision_scores = model.decision_function(X_scaled)
            
            # 결과 정리
            outliers = outlier_labels == -1
            outlier_indices = np.where(outliers)[0].tolist()
            outlier_data = X[outliers].tolist()
            
            return {
                "method": "one_class_svm",
                "contamination": contamination,
                "outlier_indices": outlier_indices,
                "outlier_data": outlier_data,
                "outlier_count": np.sum(outliers),
                "outlier_percentage": (np.sum(outliers) / len(data)) * 100,
                "decision_scores": decision_scores.tolist(),
                "interpretation": f"One-Class SVM으로 {np.sum(outliers)}개의 이상치를 탐지했습니다 ({(np.sum(outliers)/len(data)*100):.1f}%)"
            }
            
        except ImportError:
            return {"error": "scikit-learn 라이브러리가 필요합니다."}
        except Exception as e:
            return {"error": f"One-Class SVM 탐지 중 오류: {str(e)}"}
    
    @staticmethod
    def ensemble_detection(data: List[List[float]], contamination: float = 0.1) -> Dict[str, Any]:
        """앙상블 이상 탐지"""
        try:
            if not data or len(data) < 5:
                return {"error": "최소 5개 이상의 데이터 포인트가 필요합니다."}
            
            # 각 방법으로 이상치 탐지
            methods_results = {}
            
            # Isolation Forest
            iso_result = AnomalyDetector.isolation_forest_detection(data, contamination)
            if "error" not in iso_result:
                methods_results["isolation_forest"] = iso_result["outlier_indices"]
            
            # LOF
            lof_result = AnomalyDetector.local_outlier_factor(data, contamination)
            if "error" not in lof_result:
                methods_results["lof"] = lof_result["outlier_indices"]
            
            # One-Class SVM
            svm_result = AnomalyDetector.one_class_svm(data, contamination)
            if "error" not in svm_result:
                methods_results["one_class_svm"] = svm_result["outlier_indices"]
            
            if not methods_results:
                return {"error": "모든 방법이 실패했습니다."}
            
            # 투표 기반 앙상블
            all_indices = set()
            for indices in methods_results.values():
                all_indices.update(indices)
            
            # 각 인덱스에 대한 투표 수 계산
            vote_counts = {}
            for idx in all_indices:
                vote_counts[idx] = sum(1 for indices in methods_results.values() if idx in indices)
            
            # 과반수 투표
            majority_threshold = len(methods_results) / 2
            majority_outliers = [idx for idx, count in vote_counts.items() if count > majority_threshold]
            
            # 만장일치 투표
            unanimous_outliers = [idx for idx, count in vote_counts.items() if count == len(methods_results)]
            
            return {
                "method": "ensemble",
                "methods_used": list(methods_results.keys()),
                "contamination": contamination,
                "majority_vote": {
                    "outlier_indices": majority_outliers,
                    "outlier_count": len(majority_outliers),
                    "outlier_percentage": (len(majority_outliers) / len(data)) * 100
                },
                "unanimous_vote": {
                    "outlier_indices": unanimous_outliers,
                    "outlier_count": len(unanimous_outliers),
                    "outlier_percentage": (len(unanimous_outliers) / len(data)) * 100
                },
                "vote_counts": vote_counts,
                "individual_results": {
                    "isolation_forest": iso_result,
                    "lof": lof_result,
                    "one_class_svm": svm_result
                },
                "interpretation": f"앙상블 방법으로 과반수 투표: {len(majority_outliers)}개, 만장일치: {len(unanimous_outliers)}개의 이상치를 탐지했습니다."
            }
            
        except Exception as e:
            return {"error": f"앙상블 탐지 중 오류: {str(e)}"}
    
    @staticmethod
    def timeseries_anomalies(data: List[float], window_size: int = 20, threshold: float = 3.0) -> Dict[str, Any]:
        """시계열 이상 탐지"""
        try:
            if len(data) < window_size * 2:
                return {"error": f"시계열 이상 탐지를 위해서는 최소 {window_size * 2}개의 데이터 포인트가 필요합니다."}
            
            series = pd.Series(data)
            
            # 이동평균 및 표준편차 계산
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()
            
            # 이동평균에서 벗어난 정도 계산
            deviations = np.abs(series - rolling_mean) / rolling_std
            outlier_mask = deviations > threshold
            
            outlier_indices = series[outlier_mask].index.tolist()
            outlier_values = series[outlier_mask].tolist()
            
            # 연속된 이상치 구간 찾기
            anomaly_segments = []
            if outlier_indices:
                start_idx = outlier_indices[0]
                prev_idx = outlier_indices[0]
                
                for i, idx in enumerate(outlier_indices[1:], 1):
                    if idx - prev_idx > 1:  # 연속되지 않음
                        anomaly_segments.append({
                            "start": start_idx,
                            "end": prev_idx,
                            "length": prev_idx - start_idx + 1
                        })
                        start_idx = idx
                    prev_idx = idx
                
                # 마지막 구간 추가
                anomaly_segments.append({
                    "start": start_idx,
                    "end": prev_idx,
                    "length": prev_idx - start_idx + 1
                })
            
            return {
                "method": "timeseries_moving_average",
                "window_size": window_size,
                "threshold": threshold,
                "outlier_indices": outlier_indices,
                "outlier_values": outlier_values,
                "outlier_count": len(outlier_indices),
                "outlier_percentage": (len(outlier_indices) / len(data)) * 100,
                "deviations": deviations.fillna(0).tolist(),
                "rolling_mean": rolling_mean.fillna(series.mean()).tolist(),
                "rolling_std": rolling_std.fillna(series.std()).tolist(),
                "anomaly_segments": anomaly_segments,
                "interpretation": f"시계열 이상 탐지로 {len(outlier_indices)}개의 이상치를 {len(anomaly_segments)}개 구간에서 탐지했습니다."
            }
            
        except Exception as e:
            return {"error": f"시계열 이상 탐지 중 오류: {str(e)}"}
    
    @staticmethod
    def seasonal_anomalies(data: List[float], period: int = 12) -> Dict[str, Any]:
        """계절성 기반 이상 탐지"""
        try:
            if len(data) < period * 2:
                return {"error": f"계절성 이상 탐지를 위해서는 최소 {period * 2}개의 데이터 포인트가 필요합니다."}
            
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            series = pd.Series(data)
            
            # 계절성 분해
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            # 잔차(residual)에서 이상치 탐지
            residuals = decomposition.resid.dropna()
            residual_mean = residuals.mean()
            residual_std = residuals.std()
            
            threshold = 3.0
            outlier_mask = np.abs(residuals - residual_mean) > threshold * residual_std
            
            outlier_indices = residuals[outlier_mask].index.tolist()
            outlier_values = series.iloc[outlier_indices].tolist()
            
            return {
                "method": "seasonal_decomposition",
                "period": period,
                "threshold": threshold,
                "outlier_indices": outlier_indices,
                "outlier_values": outlier_values,
                "outlier_count": len(outlier_indices),
                "outlier_percentage": (len(outlier_indices) / len(data)) * 100,
                "trend": decomposition.trend.dropna().tolist(),
                "seasonal": decomposition.seasonal.tolist(),
                "residuals": residuals.tolist(),
                "residual_statistics": {
                    "mean": residual_mean,
                    "std": residual_std,
                    "threshold_value": threshold * residual_std
                },
                "interpretation": f"계절성 분해 기반으로 주기 {period}에서 {len(outlier_indices)}개의 이상치를 탐지했습니다."
            }
            
        except ImportError:
            return {"error": "statsmodels 라이브러리가 필요합니다."}
        except Exception as e:
            return {"error": f"계절성 이상 탐지 중 오류: {str(e)}"}


# MCP 도구 등록
@mcp.tool
def statistical_outliers(data: List[float], method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
    """
    통계적 방법으로 이상치를 탐지합니다.
    
    Args:
        data: 분석할 데이터 (숫자 리스트)
        method: 탐지 방법 (zscore, iqr, modified_zscore)
        threshold: 이상치 임계값
    
    Returns:
        이상치 탐지 결과
    """
    return AnomalyDetector.statistical_outliers(data, method, threshold)


@mcp.tool
def isolation_forest_detection(data: List[List[float]], contamination: float = 0.1) -> Dict[str, Any]:
    """
    Isolation Forest를 사용한 이상치 탐지입니다.
    
    Args:
        data: 분석할 데이터 (2차원 리스트 또는 1차원 리스트)
        contamination: 예상 이상치 비율 (0.0-0.5)
    
    Returns:
        이상치 탐지 결과
    """
    return AnomalyDetector.isolation_forest_detection(data, contamination)


@mcp.tool
def local_outlier_factor(data: List[List[float]], contamination: float = 0.1, n_neighbors: int = 20) -> Dict[str, Any]:
    """
    Local Outlier Factor를 사용한 이상치 탐지입니다.
    
    Args:
        data: 분석할 데이터
        contamination: 예상 이상치 비율
        n_neighbors: 이웃 수
    
    Returns:
        이상치 탐지 결과
    """
    return AnomalyDetector.local_outlier_factor(data, contamination, n_neighbors)


@mcp.tool
def one_class_svm(data: List[List[float]], contamination: float = 0.1) -> Dict[str, Any]:
    """
    One-Class SVM을 사용한 이상치 탐지입니다.
    
    Args:
        data: 분석할 데이터
        contamination: 예상 이상치 비율
    
    Returns:
        이상치 탐지 결과
    """
    return AnomalyDetector.one_class_svm(data, contamination)


@mcp.tool
def ensemble_detection(data: List[List[float]], contamination: float = 0.1) -> Dict[str, Any]:
    """
    여러 방법을 조합한 앙상블 이상치 탐지입니다.
    
    Args:
        data: 분석할 데이터
        contamination: 예상 이상치 비율
    
    Returns:
        앙상블 이상치 탐지 결과
    """
    return AnomalyDetector.ensemble_detection(data, contamination)


@mcp.tool
def timeseries_anomalies(data: List[float], window_size: int = 20, threshold: float = 3.0) -> Dict[str, Any]:
    """
    시계열 데이터의 이상치를 탐지합니다.
    
    Args:
        data: 시계열 데이터
        window_size: 이동평균 윈도우 크기
        threshold: 이상치 임계값
    
    Returns:
        시계열 이상치 탐지 결과
    """
    return AnomalyDetector.timeseries_anomalies(data, window_size, threshold)


@mcp.tool
def seasonal_anomalies(data: List[float], period: int = 12) -> Dict[str, Any]:
    """
    계절성을 고려한 시계열 이상치 탐지입니다.
    
    Args:
        data: 시계열 데이터
        period: 계절성 주기
    
    Returns:
        계절성 기반 이상치 탐지 결과
    """
    return AnomalyDetector.seasonal_anomalies(data, period)


@mcp.tool
def comprehensive_anomaly_analysis(data: List[List[float]], 
                                 contamination: float = 0.1,
                                 include_timeseries: bool = False,
                                 timeseries_column: int = 0) -> Dict[str, Any]:
    """
    종합적인 이상치 분석을 수행합니다.
    
    Args:
        data: 분석할 데이터
        contamination: 예상 이상치 비율
        include_timeseries: 시계열 분석 포함 여부
        timeseries_column: 시계열 분석할 컬럼 인덱스
    
    Returns:
        종합 이상치 분석 결과
    """
    try:
        if not data or len(data) < 5:
            return {"error": "최소 5개 이상의 데이터 포인트가 필요합니다."}
        
        results = {
            "analysis_summary": {
                "data_points": len(data),
                "contamination": contamination,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        # 통계적 방법들 (1차원 데이터로 변환하여 적용)
        if isinstance(data[0], (int, float)):
            # 이미 1차원 데이터
            univariate_data = data
        else:
            # 다차원 데이터의 경우 첫 번째 컬럼 사용
            univariate_data = [row[0] if isinstance(row, list) else row for row in data]
        
        results["statistical"] = {
            "zscore": AnomalyDetector.statistical_outliers(univariate_data, "zscore"),
            "iqr": AnomalyDetector.statistical_outliers(univariate_data, "iqr"),
            "modified_zscore": AnomalyDetector.statistical_outliers(univariate_data, "modified_zscore")
        }
        
        # 기계학습 방법들
        results["machine_learning"] = {
            "isolation_forest": AnomalyDetector.isolation_forest_detection(data, contamination),
            "lof": AnomalyDetector.local_outlier_factor(data, contamination),
            "one_class_svm": AnomalyDetector.one_class_svm(data, contamination)
        }
        
        # 앙상블 방법
        results["ensemble"] = AnomalyDetector.ensemble_detection(data, contamination)
        
        # 시계열 분석 (요청된 경우)
        if include_timeseries:
            if isinstance(data[0], list) and len(data[0]) > timeseries_column:
                ts_data = [row[timeseries_column] for row in data]
            else:
                ts_data = univariate_data
                
            results["timeseries"] = {
                "moving_average": AnomalyDetector.timeseries_anomalies(ts_data),
                "seasonal": AnomalyDetector.seasonal_anomalies(ts_data)
            }
        
        # 결과 요약
        outlier_counts = {}
        for category, methods in results.items():
            if category == "analysis_summary":
                continue
            
            if isinstance(methods, dict) and "error" not in methods:
                for method_name, method_result in methods.items():
                    if isinstance(method_result, dict) and "outlier_count" in method_result:
                        outlier_counts[f"{category}_{method_name}"] = method_result["outlier_count"]
        
        results["analysis_summary"]["outlier_counts"] = outlier_counts
        
        # 가장 일관된 결과 찾기
        if outlier_counts:
            avg_outliers = np.mean(list(outlier_counts.values()))
            most_consistent = min(outlier_counts.items(), key=lambda x: abs(x[1] - avg_outliers))
            results["analysis_summary"]["most_consistent_method"] = {
                "method": most_consistent[0],
                "outlier_count": most_consistent[1]
            }
        
        # 종합 해석
        total_methods = len(outlier_counts)
        if total_methods > 0:
            avg_outlier_rate = np.mean(list(outlier_counts.values())) / len(data) * 100
            results["analysis_summary"]["average_outlier_rate"] = avg_outlier_rate
            
            if avg_outlier_rate > 20:
                interpretation = "높은 이상치 비율이 감지되었습니다. 데이터 품질을 검토하세요."
            elif avg_outlier_rate > 10:
                interpretation = "중간 수준의 이상치가 탐지되었습니다. 개별 검토가 필요합니다."
            elif avg_outlier_rate > 0:
                interpretation = "일부 이상치가 탐지되었습니다. 비즈니스 영향을 평가하세요."
            else:
                interpretation = "명확한 이상치가 탐지되지 않았습니다."
                
            results["analysis_summary"]["interpretation"] = interpretation
        
        return results
        
    except Exception as e:
        return {"error": f"종합 분석 중 오류: {str(e)}"}


if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Anomaly Detection MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)