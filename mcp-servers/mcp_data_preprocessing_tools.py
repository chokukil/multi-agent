#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Tool: Data Preprocessing Tools
데이터 전처리 도구 - 결측치 처리, 이상치 제거, 스케일링, 인코딩, 특성 엔지니어링
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
import warnings
warnings.filterwarnings('ignore')

# FastMCP import
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Get port from environment variable
SERVER_PORT = int(os.getenv('SERVER_PORT', '8017'))

# FastMCP 서버 생성
mcp = FastMCP("Data Preprocessing Tools")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessingTools:
    """데이터 전처리 도구 클래스"""
    
    @staticmethod
    def handle_missing_values(data: List[List[Union[float, str, None]]], 
                            column_names: List[str] = None,
                            method: str = "auto",
                            target_columns: List[str] = None) -> Dict[str, Any]:
        """결측치 처리"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            original_shape = df.shape
            processing_log = []
            
            # 결측치 분석
            missing_info = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                missing_info[col] = {
                    "count": missing_count,
                    "percentage": missing_pct
                }
            
            # 처리할 컬럼 결정
            if target_columns:
                cols_to_process = [col for col in target_columns if col in df.columns]
            else:
                cols_to_process = [col for col in df.columns if df[col].isnull().any()]
            
            # 컬럼별 처리
            for col in cols_to_process:
                missing_count = missing_info[col]["count"]
                missing_pct = missing_info[col]["percentage"]
                
                if missing_count == 0:
                    continue
                
                # 전략 결정
                if method == "auto":
                    if missing_pct > 50:
                        strategy = "drop_column"
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        strategy = "fill_median" if missing_pct > 20 else "fill_mean"
                    else:
                        strategy = "fill_mode"
                else:
                    strategy = method
                
                # 결측치 처리 실행
                if strategy == "drop_column":
                    df = df.drop(columns=[col])
                    processing_log.append(f"컬럼 '{col}' 삭제 (결측치 {missing_pct:.1f}%)")
                
                elif strategy == "fill_mean":
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    processing_log.append(f"컬럼 '{col}': 평균값({mean_val:.2f})으로 {missing_count}개 채움")
                
                elif strategy == "fill_median":
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    processing_log.append(f"컬럼 '{col}': 중앙값({median_val:.2f})으로 {missing_count}개 채움")
                
                elif strategy == "fill_mode":
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    processing_log.append(f"컬럼 '{col}': 최빈값('{mode_val}')으로 {missing_count}개 채움")
                
                elif strategy == "forward_fill":
                    df[col] = df[col].fillna(method='ffill')
                    processing_log.append(f"컬럼 '{col}': 이전 값으로 {missing_count}개 채움")
                
                elif strategy == "backward_fill":
                    df[col] = df[col].fillna(method='bfill')
                    processing_log.append(f"컬럼 '{col}': 다음 값으로 {missing_count}개 채움")
                
                elif strategy == "drop_rows":
                    initial_rows = len(df)
                    df = df.dropna(subset=[col])
                    rows_dropped = initial_rows - len(df)
                    processing_log.append(f"컬럼 '{col}': 결측치 행 {rows_dropped}개 삭제")
            
            # 결과 정리
            processed_data = df.values.tolist()
            processed_columns = df.columns.tolist()
            
            return {
                "processed_data": processed_data,
                "column_names": processed_columns,
                "original_shape": original_shape,
                "processed_shape": df.shape,
                "missing_info_before": missing_info,
                "processing_log": processing_log,
                "missing_remaining": df.isnull().sum().sum(),
                "interpretation": f"결측치 처리 완료: {original_shape} → {df.shape}, {len(processing_log)}개 작업 수행"
            }
            
        except Exception as e:
            return {"error": f"결측치 처리 중 오류: {str(e)}"}
    
    @staticmethod
    def detect_and_remove_outliers(data: List[List[float]], 
                                 column_names: List[str] = None,
                                 method: str = "iqr",
                                 threshold: float = 1.5) -> Dict[str, Any]:
        """이상치 탐지 및 제거"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            original_shape = df.shape
            outlier_info = {}
            processing_log = []
            
            # 수치형 컬럼만 처리
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                col_data = df[col].dropna()
                
                if method == "iqr":
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    outlier_info[col] = {
                        "method": "IQR",
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "outlier_count": outlier_count,
                        "outlier_percentage": (outlier_count / len(df)) * 100
                    }
                
                elif method == "zscore":
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    outlier_mask = z_scores > threshold
                    outlier_count = outlier_mask.sum()
                    
                    outlier_info[col] = {
                        "method": "Z-Score",
                        "threshold": threshold,
                        "outlier_count": outlier_count,
                        "outlier_percentage": (outlier_count / len(df)) * 100
                    }
                
                elif method == "isolation_forest":
                    try:
                        from sklearn.ensemble import IsolationForest
                        
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                        outlier_mask = outlier_labels == -1
                        outlier_count = outlier_mask.sum()
                        
                        outlier_info[col] = {
                            "method": "Isolation Forest",
                            "contamination": 0.1,
                            "outlier_count": outlier_count,
                            "outlier_percentage": (outlier_count / len(df)) * 100
                        }
                    
                    except ImportError:
                        # sklearn이 없으면 IQR 방법 사용
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                        outlier_count = outlier_mask.sum()
                        
                        outlier_info[col] = {
                            "method": "IQR (fallback)",
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "outlier_count": outlier_count,
                            "outlier_percentage": (outlier_count / len(df)) * 100
                        }
                
                # 이상치 제거
                if outlier_count > 0:
                    df = df[~outlier_mask]
                    processing_log.append(f"컬럼 '{col}': {method} 방법으로 {outlier_count}개 이상치 제거")
            
            processed_data = df.values.tolist()
            
            return {
                "processed_data": processed_data,
                "column_names": df.columns.tolist(),
                "original_shape": original_shape,
                "processed_shape": df.shape,
                "outlier_info": outlier_info,
                "processing_log": processing_log,
                "total_outliers_removed": original_shape[0] - df.shape[0],
                "interpretation": f"이상치 제거 완료: {original_shape} → {df.shape}, {len(numeric_columns)}개 컬럼 처리"
            }
            
        except Exception as e:
            return {"error": f"이상치 처리 중 오류: {str(e)}"}
    
    @staticmethod
    def scale_numerical_features(data: List[List[float]], 
                               column_names: List[str] = None,
                               method: str = "standard",
                               target_columns: List[str] = None) -> Dict[str, Any]:
        """수치형 특성 스케일링"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            original_data = df.copy()
            processing_log = []
            scaling_info = {}
            
            # 처리할 컬럼 결정
            if target_columns:
                numeric_columns = [col for col in target_columns 
                                 if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                return {"error": "스케일링할 수치형 컬럼이 없습니다."}
            
            # 스케일링 수행
            if method == "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                
                scaling_info = {
                    "method": "Standard (Z-score)",
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "columns": numeric_columns
                }
                processing_log.append(f"표준화 스케일링 완료: {len(numeric_columns)}개 컬럼")
            
            elif method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                
                scaling_info = {
                    "method": "Min-Max (0-1)",
                    "min": scaler.data_min_.tolist(),
                    "max": scaler.data_max_.tolist(),
                    "columns": numeric_columns
                }
                processing_log.append(f"Min-Max 스케일링 완료: {len(numeric_columns)}개 컬럼")
            
            elif method == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                
                scaling_info = {
                    "method": "Robust (Median-IQR)",
                    "center": scaler.center_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "columns": numeric_columns
                }
                processing_log.append(f"로버스트 스케일링 완료: {len(numeric_columns)}개 컬럼")
            
            elif method == "normalize":
                from sklearn.preprocessing import normalize
                df[numeric_columns] = normalize(df[numeric_columns], axis=0)
                
                scaling_info = {
                    "method": "L2 Normalization",
                    "columns": numeric_columns
                }
                processing_log.append(f"정규화 완료: {len(numeric_columns)}개 컬럼")
            
            # 스케일링 전후 통계 비교
            before_stats = {}
            after_stats = {}
            
            for col in numeric_columns:
                before_stats[col] = {
                    "mean": float(original_data[col].mean()),
                    "std": float(original_data[col].std()),
                    "min": float(original_data[col].min()),
                    "max": float(original_data[col].max())
                }
                after_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
            
            return {
                "processed_data": df.values.tolist(),
                "column_names": df.columns.tolist(),
                "scaling_info": scaling_info,
                "processing_log": processing_log,
                "statistics_before": before_stats,
                "statistics_after": after_stats,
                "columns_scaled": numeric_columns,
                "interpretation": f"{method} 스케일링 완료: {len(numeric_columns)}개 수치형 컬럼 처리"
            }
            
        except Exception as e:
            return {"error": f"스케일링 중 오류: {str(e)}"}
    
    @staticmethod
    def encode_categorical_features(data: List[List[Union[str, float]]], 
                                  column_names: List[str] = None,
                                  method: str = "auto",
                                  target_columns: List[str] = None) -> Dict[str, Any]:
        """범주형 특성 인코딩"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            original_columns = df.columns.tolist()
            processing_log = []
            encoding_info = {}
            
            # 처리할 컬럼 결정
            if target_columns:
                categorical_columns = [col for col in target_columns 
                                     if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
            else:
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_columns:
                return {"error": "인코딩할 범주형 컬럼이 없습니다."}
            
            for col in categorical_columns:
                unique_count = df[col].nunique()
                
                # 인코딩 방법 결정
                if method == "auto":
                    if unique_count == 2:
                        encoding_method = "label"
                    elif unique_count <= 10:
                        encoding_method = "onehot"
                    else:
                        encoding_method = "label"
                else:
                    encoding_method = method
                
                # 인코딩 실행
                if encoding_method == "onehot":
                    # One-hot 인코딩
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                    df = df.drop(columns=[col])
                    df = pd.concat([df, dummies], axis=1)
                    
                    encoding_info[col] = {
                        "method": "One-Hot",
                        "original_categories": df[col].unique().tolist() if col in df.columns else [],
                        "encoded_columns": dummies.columns.tolist(),
                        "unique_count": unique_count
                    }
                    processing_log.append(f"컬럼 '{col}': One-Hot 인코딩 ({len(dummies.columns)}개 컬럼 생성)")
                
                elif encoding_method == "label":
                    # Label 인코딩
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    original_categories = df[col].unique().tolist()
                    df[col] = le.fit_transform(df[col].astype(str))
                    
                    encoding_info[col] = {
                        "method": "Label",
                        "original_categories": original_categories,
                        "label_mapping": dict(zip(le.classes_, le.transform(le.classes_))),
                        "unique_count": unique_count
                    }
                    processing_log.append(f"컬럼 '{col}': Label 인코딩 ({unique_count}개 범주)")
                
                elif encoding_method == "ordinal":
                    # 서수 인코딩 (간단한 버전)
                    categories = df[col].unique()
                    category_mapping = {cat: i for i, cat in enumerate(categories)}
                    df[col] = df[col].map(category_mapping)
                    
                    encoding_info[col] = {
                        "method": "Ordinal",
                        "category_mapping": category_mapping,
                        "unique_count": unique_count
                    }
                    processing_log.append(f"컬럼 '{col}': 서수 인코딩 ({unique_count}개 범주)")
            
            return {
                "processed_data": df.values.tolist(),
                "column_names": df.columns.tolist(),
                "original_columns": original_columns,
                "encoding_info": encoding_info,
                "processing_log": processing_log,
                "columns_added": len(df.columns) - len(original_columns),
                "categorical_columns_processed": categorical_columns,
                "interpretation": f"범주형 인코딩 완료: {len(categorical_columns)}개 컬럼 처리, {len(df.columns) - len(original_columns)}개 컬럼 추가"
            }
            
        except Exception as e:
            return {"error": f"인코딩 중 오류: {str(e)}"}
    
    @staticmethod
    def create_polynomial_features(data: List[List[float]], 
                                 column_names: List[str] = None,
                                 degree: int = 2,
                                 interaction_only: bool = False,
                                 include_bias: bool = False) -> Dict[str, Any]:
        """다항식 특성 생성"""
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            # 수치형 컬럼만 처리
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                return {"error": "다항식 특성 생성을 위한 수치형 컬럼이 없습니다."}
            
            if len(numeric_columns) > 10:
                return {"error": "너무 많은 특성으로 인해 차원이 폭발할 수 있습니다. 10개 이하의 특성을 사용하세요."}
            
            # 다항식 특성 생성
            poly = PolynomialFeatures(
                degree=degree, 
                interaction_only=interaction_only,
                include_bias=include_bias
            )
            
            numeric_data = df[numeric_columns]
            poly_features = poly.fit_transform(numeric_data)
            
            # 특성 이름 생성
            feature_names = poly.get_feature_names_out(numeric_columns)
            
            # 원본 데이터와 결합
            non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
            result_df = df[non_numeric_columns].copy() if non_numeric_columns else pd.DataFrame()
            
            # 다항식 특성 추가
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
            result_df = pd.concat([result_df, poly_df], axis=1)
            
            return {
                "processed_data": result_df.values.tolist(),
                "column_names": result_df.columns.tolist(),
                "original_features": len(numeric_columns),
                "polynomial_features": len(feature_names),
                "degree": degree,
                "interaction_only": interaction_only,
                "include_bias": include_bias,
                "feature_names": feature_names.tolist(),
                "processing_info": {
                    "original_columns": numeric_columns,
                    "features_added": len(feature_names) - len(numeric_columns)
                },
                "interpretation": f"다항식 특성 생성 완료: {len(numeric_columns)}개 → {len(feature_names)}개 특성 ({degree}차)"
            }
            
        except Exception as e:
            return {"error": f"다항식 특성 생성 중 오류: {str(e)}"}
    
    @staticmethod
    def remove_duplicates(data: List[List[Any]], 
                        column_names: List[str] = None,
                        subset_columns: List[str] = None,
                        keep: str = "first") -> Dict[str, Any]:
        """중복 데이터 제거"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            original_shape = df.shape
            
            # 중복 분석
            duplicate_count = df.duplicated(subset=subset_columns, keep=False).sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100
            
            # 중복 제거
            df_clean = df.drop_duplicates(subset=subset_columns, keep=keep)
            removed_count = len(df) - len(df_clean)
            
            # 중복 패턴 분석
            if subset_columns:
                duplicate_analysis = df[df.duplicated(subset=subset_columns, keep=False)].groupby(subset_columns).size()
            else:
                duplicate_analysis = df[df.duplicated(keep=False)].groupby(list(df.columns)).size()
            
            top_duplicates = duplicate_analysis.head(10).to_dict() if not duplicate_analysis.empty else {}
            
            return {
                "processed_data": df_clean.values.tolist(),
                "column_names": df_clean.columns.tolist(),
                "original_shape": original_shape,
                "processed_shape": df_clean.shape,
                "duplicates_found": duplicate_count,
                "duplicates_removed": removed_count,
                "duplicate_percentage": duplicate_percentage,
                "subset_columns": subset_columns,
                "keep_strategy": keep,
                "top_duplicate_patterns": top_duplicates,
                "interpretation": f"중복 제거 완료: {removed_count}개 중복 행 제거 ({original_shape} → {df_clean.shape})"
            }
            
        except Exception as e:
            return {"error": f"중복 제거 중 오류: {str(e)}"}


# MCP 도구 등록
@mcp.tool("handle_missing_values")
def handle_missing_values(data: List[List[Union[float, str, None]]], 
                        column_names: List[str] = None,
                        method: str = "auto",
                        target_columns: List[str] = None) -> Dict[str, Any]:
    """
    결측치를 처리합니다.
    
    Args:
        data: 처리할 데이터 (2차원 리스트)
        column_names: 컬럼 이름들
        method: 처리 방법 (auto, fill_mean, fill_median, fill_mode, forward_fill, backward_fill, drop_rows, drop_column)
        target_columns: 처리할 특정 컬럼들
    
    Returns:
        결측치 처리 결과
    """
    return DataPreprocessingTools.handle_missing_values(data, column_names, method, target_columns)


@mcp.tool("detect_and_remove_outliers")
def detect_and_remove_outliers(data: List[List[float]], 
                             column_names: List[str] = None,
                             method: str = "iqr",
                             threshold: float = 1.5) -> Dict[str, Any]:
    """
    이상치를 탐지하고 제거합니다.
    
    Args:
        data: 처리할 수치 데이터
        column_names: 컬럼 이름들
        method: 탐지 방법 (iqr, zscore, isolation_forest)
        threshold: 임계값
    
    Returns:
        이상치 탐지 및 제거 결과
    """
    return DataPreprocessingTools.detect_and_remove_outliers(data, column_names, method, threshold)


@mcp.tool("scale_numerical_features")
def scale_numerical_features(data: List[List[float]], 
                           column_names: List[str] = None,
                           method: str = "standard",
                           target_columns: List[str] = None) -> Dict[str, Any]:
    """
    수치형 특성을 스케일링합니다.
    
    Args:
        data: 스케일링할 수치 데이터
        column_names: 컬럼 이름들
        method: 스케일링 방법 (standard, minmax, robust, normalize)
        target_columns: 스케일링할 특정 컬럼들
    
    Returns:
        스케일링 결과
    """
    return DataPreprocessingTools.scale_numerical_features(data, column_names, method, target_columns)


@mcp.tool("encode_categorical_features")
def encode_categorical_features(data: List[List[Union[str, float]]], 
                              column_names: List[str] = None,
                              method: str = "auto",
                              target_columns: List[str] = None) -> Dict[str, Any]:
    """
    범주형 특성을 인코딩합니다.
    
    Args:
        data: 인코딩할 데이터
        column_names: 컬럼 이름들
        method: 인코딩 방법 (auto, onehot, label, ordinal)
        target_columns: 인코딩할 특정 컬럼들
    
    Returns:
        인코딩 결과
    """
    return DataPreprocessingTools.encode_categorical_features(data, column_names, method, target_columns)


@mcp.tool("create_polynomial_features")
def create_polynomial_features(data: List[List[float]], 
                             column_names: List[str] = None,
                             degree: int = 2,
                             interaction_only: bool = False,
                             include_bias: bool = False) -> Dict[str, Any]:
    """
    다항식 특성을 생성합니다.
    
    Args:
        data: 특성 생성할 수치 데이터
        column_names: 컬럼 이름들
        degree: 다항식 차수
        interaction_only: 상호작용 항만 생성 여부
        include_bias: 절편 포함 여부
    
    Returns:
        다항식 특성 생성 결과
    """
    return DataPreprocessingTools.create_polynomial_features(data, column_names, degree, interaction_only, include_bias)


@mcp.tool("remove_duplicates")
def remove_duplicates(data: List[List[Any]], 
                    column_names: List[str] = None,
                    subset_columns: List[str] = None,
                    keep: str = "first") -> Dict[str, Any]:
    """
    중복 데이터를 제거합니다.
    
    Args:
        data: 처리할 데이터
        column_names: 컬럼 이름들
        subset_columns: 중복 검사할 특정 컬럼들
        keep: 중복 유지 전략 (first, last, False)
    
    Returns:
        중복 제거 결과
    """
    return DataPreprocessingTools.remove_duplicates(data, column_names, subset_columns, keep)


@mcp.tool("comprehensive_preprocessing")
def comprehensive_preprocessing(data: List[List[Any]], 
                              column_names: List[str] = None,
                              operations: List[str] = None,
                              missing_method: str = "auto",
                              outlier_method: str = "iqr",
                              scaling_method: str = "standard",
                              encoding_method: str = "auto") -> Dict[str, Any]:
    """
    종합적인 데이터 전처리를 수행합니다.
    
    Args:
        data: 처리할 데이터
        column_names: 컬럼 이름들
        operations: 수행할 작업 리스트 (duplicates, missing, outliers, scaling, encoding)
        missing_method: 결측치 처리 방법
        outlier_method: 이상치 처리 방법
        scaling_method: 스케일링 방법
        encoding_method: 인코딩 방법
    
    Returns:
        종합 전처리 결과
    """
    try:
        if operations is None:
            operations = ["duplicates", "missing", "outliers", "encoding", "scaling"]
        
        current_data = data
        current_columns = column_names
        processing_steps = []
        overall_log = []
        
        # 1. 중복 제거
        if "duplicates" in operations:
            result = DataPreprocessingTools.remove_duplicates(current_data, current_columns)
            if "error" not in result:
                current_data = result["processed_data"]
                current_columns = result["column_names"]
                processing_steps.append("중복 제거")
                overall_log.extend([f"중복 제거: {result['duplicates_removed']}개 행 제거"])
        
        # 2. 결측치 처리
        if "missing" in operations:
            result = DataPreprocessingTools.handle_missing_values(
                current_data, current_columns, missing_method
            )
            if "error" not in result:
                current_data = result["processed_data"]
                current_columns = result["column_names"]
                processing_steps.append("결측치 처리")
                overall_log.extend(result["processing_log"])
        
        # 3. 이상치 처리 (수치형 데이터가 있는 경우)
        if "outliers" in operations:
            try:
                result = DataPreprocessingTools.detect_and_remove_outliers(
                    current_data, current_columns, outlier_method
                )
                if "error" not in result:
                    current_data = result["processed_data"]
                    current_columns = result["column_names"]
                    processing_steps.append("이상치 처리")
                    overall_log.extend(result["processing_log"])
            except:
                overall_log.append("이상치 처리 건너뜀 (수치형 데이터 없음)")
        
        # 4. 범주형 인코딩
        if "encoding" in operations:
            try:
                result = DataPreprocessingTools.encode_categorical_features(
                    current_data, current_columns, encoding_method
                )
                if "error" not in result:
                    current_data = result["processed_data"]
                    current_columns = result["column_names"]
                    processing_steps.append("범주형 인코딩")
                    overall_log.extend(result["processing_log"])
            except:
                overall_log.append("인코딩 건너뜀 (범주형 데이터 없음)")
        
        # 5. 스케일링
        if "scaling" in operations:
            try:
                result = DataPreprocessingTools.scale_numerical_features(
                    current_data, current_columns, scaling_method
                )
                if "error" not in result:
                    current_data = result["processed_data"]
                    current_columns = result["column_names"]
                    processing_steps.append("스케일링")
                    overall_log.extend(result["processing_log"])
            except:
                overall_log.append("스케일링 건너뜀 (수치형 데이터 없음)")
        
        return {
            "processed_data": current_data,
            "column_names": current_columns,
            "original_shape": (len(data), len(data[0]) if data else 0),
            "final_shape": (len(current_data), len(current_data[0]) if current_data else 0),
            "processing_steps": processing_steps,
            "detailed_log": overall_log,
            "operations_completed": len(processing_steps),
            "interpretation": f"종합 전처리 완료: {len(processing_steps)}개 단계 수행"
        }
        
    except Exception as e:
        return {"error": f"종합 전처리 중 오류: {str(e)}"}


if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Data Preprocessing Tools MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)