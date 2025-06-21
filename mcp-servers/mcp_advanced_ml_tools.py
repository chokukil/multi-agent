#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Tool: Advanced ML Tools
고급 머신러닝 도구 - 모델 구축, 하이퍼파라미터 튜닝, 모델 평가 및 해석
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
SERVER_PORT = int(os.getenv('SERVER_PORT', '8016'))

# FastMCP 서버 생성
mcp = FastMCP("Advanced ML Tools")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMLTools:
    """고급 머신러닝 도구 클래스"""
    
    @staticmethod
    def build_classification_model(data: List[List[float]], 
                                 target: List[Union[int, str]], 
                                 feature_names: List[str] = None,
                                 algorithm: str = "auto",
                                 test_size: float = 0.2) -> Dict[str, Any]:
        """분류 모델 구축"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.preprocessing import LabelEncoder
            
            # 데이터 변환
            X = np.array(data)
            y = np.array(target)
            
            if len(X) < 10:
                return {"error": "분류 모델 구축을 위해서는 최소 10개 이상의 데이터가 필요합니다."}
            
            # 타겟 인코딩 (문자열인 경우)
            label_encoder = None
            if not np.issubdtype(y.dtype, np.number):
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            else:
                y_encoded = y
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            # 모델 선택 및 훈련
            models = {}
            
            if algorithm == "auto" or algorithm == "random_forest":
                models["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42)
            
            if algorithm == "auto" or algorithm == "gradient_boosting":
                models["GradientBoosting"] = GradientBoostingClassifier(random_state=42)
            
            if algorithm == "auto" or algorithm == "svm":
                models["SVM"] = SVC(probability=True, random_state=42)
            
            if algorithm == "auto" or algorithm == "logistic":
                models["LogisticRegression"] = LogisticRegression(random_state=42, max_iter=1000)
            
            # XGBoost 추가 (사용 가능한 경우)
            if algorithm == "auto" or algorithm == "xgboost":
                try:
                    import xgboost as xgb
                    models["XGBoost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                except ImportError:
                    pass
            
            # 각 모델 훈련 및 평가
            results = {}
            best_model = None
            best_score = 0
            best_name = ""
            
            for name, model in models.items():
                try:
                    # 훈련
                    model.fit(X_train, y_train)
                    
                    # 예측
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    # 성능 평가
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # 분류 보고서
                    target_names = label_encoder.classes_ if label_encoder else None
                    class_report = classification_report(
                        y_test, y_pred, 
                        target_names=target_names, 
                        output_dict=True, 
                        zero_division=0
                    )
                    
                    # 혼동 행렬
                    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
                    
                    # 특성 중요도 (가능한 경우)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        if feature_names:
                            feature_importance = dict(zip(feature_names, model.feature_importances_))
                        else:
                            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
                    
                    results[name] = {
                        "accuracy": accuracy,
                        "classification_report": class_report,
                        "confusion_matrix": conf_matrix,
                        "feature_importance": feature_importance,
                        "predictions": y_pred.tolist(),
                        "probabilities": y_proba.tolist() if y_proba is not None else None
                    }
                    
                    # 최고 모델 추적
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = name
                        best_name = name
                        
                except Exception as e:
                    results[name] = {"error": str(e)}
            
            return {
                "task_type": "classification",
                "best_model": best_name,
                "best_accuracy": best_score,
                "models": results,
                "data_info": {
                    "total_samples": len(X),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X.shape[1],
                    "classes": len(np.unique(y_encoded))
                },
                "label_encoder_classes": label_encoder.classes_.tolist() if label_encoder else None,
                "interpretation": f"총 {len(models)}개 모델 중 {best_name}이 {best_score:.3f} 정확도로 최고 성능을 보였습니다."
            }
            
        except Exception as e:
            return {"error": f"분류 모델 구축 중 오류: {str(e)}"}
    
    @staticmethod
    def build_regression_model(data: List[List[float]], 
                             target: List[float], 
                             feature_names: List[str] = None,
                             algorithm: str = "auto",
                             test_size: float = 0.2) -> Dict[str, Any]:
        """회귀 모델 구축"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # 데이터 변환
            X = np.array(data)
            y = np.array(target)
            
            if len(X) < 10:
                return {"error": "회귀 모델 구축을 위해서는 최소 10개 이상의 데이터가 필요합니다."}
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # 모델 선택 및 훈련
            models = {}
            
            if algorithm == "auto" or algorithm == "random_forest":
                models["RandomForest"] = RandomForestRegressor(n_estimators=100, random_state=42)
            
            if algorithm == "auto" or algorithm == "gradient_boosting":
                models["GradientBoosting"] = GradientBoostingRegressor(random_state=42)
            
            if algorithm == "auto" or algorithm == "svr":
                models["SVR"] = SVR()
            
            if algorithm == "auto" or algorithm == "linear":
                models["LinearRegression"] = LinearRegression()
            
            if algorithm == "auto" or algorithm == "ridge":
                models["Ridge"] = Ridge(random_state=42)
            
            if algorithm == "auto" or algorithm == "lasso":
                models["Lasso"] = Lasso(random_state=42)
            
            # XGBoost 추가 (사용 가능한 경우)
            if algorithm == "auto" or algorithm == "xgboost":
                try:
                    import xgboost as xgb
                    models["XGBoost"] = xgb.XGBRegressor(random_state=42)
                except ImportError:
                    pass
            
            # 각 모델 훈련 및 평가
            results = {}
            best_model = None
            best_score = -np.inf
            best_name = ""
            
            for name, model in models.items():
                try:
                    # 훈련
                    model.fit(X_train, y_train)
                    
                    # 예측
                    y_pred = model.predict(X_test)
                    
                    # 성능 평가
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # 특성 중요도 (가능한 경우)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        if feature_names:
                            feature_importance = dict(zip(feature_names, model.feature_importances_))
                        else:
                            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
                    elif hasattr(model, 'coef_'):
                        if feature_names:
                            feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
                        else:
                            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(np.abs(model.coef_))}
                    
                    results[name] = {
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "feature_importance": feature_importance,
                        "predictions": y_pred.tolist(),
                        "residuals": (y_test - y_pred).tolist()
                    }
                    
                    # 최고 모델 추적 (R² 기준)
                    if r2 > best_score:
                        best_score = r2
                        best_model = name
                        best_name = name
                        
                except Exception as e:
                    results[name] = {"error": str(e)}
            
            return {
                "task_type": "regression",
                "best_model": best_name,
                "best_r2": best_score,
                "models": results,
                "data_info": {
                    "total_samples": len(X),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X.shape[1],
                    "target_mean": float(np.mean(y)),
                    "target_std": float(np.std(y))
                },
                "interpretation": f"총 {len(models)}개 모델 중 {best_name}이 {best_score:.3f} R² 점수로 최고 성능을 보였습니다."
            }
            
        except Exception as e:
            return {"error": f"회귀 모델 구축 중 오류: {str(e)}"}
    
    @staticmethod
    def hyperparameter_tuning(data: List[List[float]], 
                            target: List[Union[int, str, float]], 
                            model_type: str = "random_forest",
                            task_type: str = "classification",
                            cv_folds: int = 5) -> Dict[str, Any]:
        """하이퍼파라미터 튜닝"""
        try:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.svm import SVC, SVR
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.preprocessing import LabelEncoder
            
            # 데이터 변환
            X = np.array(data)
            y = np.array(target)
            
            if len(X) < cv_folds * 2:
                return {"error": f"하이퍼파라미터 튜닝을 위해서는 최소 {cv_folds * 2}개 이상의 데이터가 필요합니다."}
            
            # 타겟 인코딩 (분류에서 문자열인 경우)
            if task_type == "classification" and not np.issubdtype(y.dtype, np.number):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            # 모델 및 파라미터 그리드 정의
            model_configs = {
                "random_forest": {
                    "classifier": RandomForestClassifier(random_state=42),
                    "regressor": RandomForestRegressor(random_state=42),
                    "params": {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                "svm": {
                    "classifier": SVC(probability=True, random_state=42),
                    "regressor": SVR(),
                    "params": {
                        'C': [0.1, 1, 10, 100],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                },
                "logistic": {
                    "classifier": LogisticRegression(random_state=42, max_iter=1000),
                    "regressor": Ridge(random_state=42),
                    "params": {
                        'C': [0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2'] if task_type == "classification" else ['l2'],
                        'solver': ['liblinear', 'lbfgs']
                    }
                }
            }
            
            if model_type not in model_configs:
                return {"error": f"지원되지 않는 모델 타입: {model_type}"}
            
            config = model_configs[model_type]
            model = config["classifier"] if task_type == "classification" else config["regressor"]
            param_grid = config["params"]
            
            # 그리드 크기에 따라 GridSearch 또는 RandomizedSearch 선택
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            
            if total_combinations > 100:
                # 조합이 많으면 RandomizedSearchCV 사용
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=50, cv=cv_folds, 
                    scoring='accuracy' if task_type == "classification" else 'r2',
                    random_state=42, n_jobs=-1
                )
                search_type = "RandomizedSearchCV"
            else:
                # 조합이 적으면 GridSearchCV 사용
                search = GridSearchCV(
                    model, param_grid, cv=cv_folds,
                    scoring='accuracy' if task_type == "classification" else 'r2',
                    n_jobs=-1
                )
                search_type = "GridSearchCV"
            
            # 하이퍼파라미터 튜닝 실행
            search.fit(X, y)
            
            # 결과 정리
            cv_results = search.cv_results_
            
            # 상위 5개 결과
            top_results = []
            for i in range(min(5, len(cv_results['mean_test_score']))):
                idx = np.argsort(cv_results['mean_test_score'])[::-1][i]
                top_results.append({
                    "rank": i + 1,
                    "params": cv_results['params'][idx],
                    "mean_score": cv_results['mean_test_score'][idx],
                    "std_score": cv_results['std_test_score'][idx]
                })
            
            return {
                "search_type": search_type,
                "model_type": model_type,
                "task_type": task_type,
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "best_estimator": str(search.best_estimator_),
                "top_results": top_results,
                "tuning_info": {
                    "cv_folds": cv_folds,
                    "total_fits": len(cv_results['mean_test_score']),
                    "parameter_combinations": total_combinations,
                    "data_samples": len(X),
                    "features": X.shape[1]
                },
                "interpretation": f"{search_type}를 사용하여 {total_combinations}개 조합 중 최적 파라미터를 찾았습니다. 최고 점수: {search.best_score_:.4f}"
            }
            
        except Exception as e:
            return {"error": f"하이퍼파라미터 튜닝 중 오류: {str(e)}"}
    
    @staticmethod
    def model_evaluation(data: List[List[float]], 
                        target: List[Union[int, str, float]], 
                        predictions: List[Union[int, str, float]],
                        task_type: str = "classification",
                        cross_validation: bool = True) -> Dict[str, Any]:
        """모델 평가"""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score, 
                confusion_matrix, classification_report, roc_auc_score
            )
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error, r2_score,
                mean_absolute_percentage_error
            )
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            import scipy.stats as stats
            
            # 데이터 변환
            X = np.array(data) if data else None
            y_true = np.array(target)
            y_pred = np.array(predictions)
            
            if len(y_true) != len(y_pred):
                return {"error": "실제값과 예측값의 길이가 다릅니다."}
            
            # 타겟 인코딩 (분류에서 문자열인 경우)
            label_encoder = None
            if task_type == "classification" and not np.issubdtype(y_true.dtype, np.number):
                label_encoder = LabelEncoder()
                y_true_encoded = label_encoder.fit_transform(y_true)
                y_pred_encoded = label_encoder.transform(y_pred)
            else:
                y_true_encoded = y_true
                y_pred_encoded = y_pred
            
            evaluation = {}
            
            if task_type == "classification":
                # 분류 평가 지표
                evaluation["accuracy"] = accuracy_score(y_true_encoded, y_pred_encoded)
                evaluation["precision"] = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
                evaluation["recall"] = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
                evaluation["f1_score"] = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
                
                # 혼동 행렬
                cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                evaluation["confusion_matrix"] = cm.tolist()
                
                # 분류 보고서
                target_names = label_encoder.classes_ if label_encoder else None
                class_report = classification_report(
                    y_true_encoded, y_pred_encoded, 
                    target_names=target_names, 
                    output_dict=True, 
                    zero_division=0
                )
                evaluation["classification_report"] = class_report
                
                # 클래스별 성능
                unique_classes = np.unique(y_true_encoded)
                class_performance = {}
                for cls in unique_classes:
                    class_name = label_encoder.inverse_transform([cls])[0] if label_encoder else str(cls)
                    tp = np.sum((y_true_encoded == cls) & (y_pred_encoded == cls))
                    fp = np.sum((y_true_encoded != cls) & (y_pred_encoded == cls))
                    fn = np.sum((y_true_encoded == cls) & (y_pred_encoded != cls))
                    tn = np.sum((y_true_encoded != cls) & (y_pred_encoded != cls))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    class_performance[class_name] = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": np.sum(y_true_encoded == cls)
                    }
                
                evaluation["class_performance"] = class_performance
                
            else:  # regression
                # 회귀 평가 지표
                evaluation["mse"] = mean_squared_error(y_true, y_pred)
                evaluation["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                evaluation["mae"] = mean_absolute_error(y_true, y_pred)
                evaluation["r2"] = r2_score(y_true, y_pred)
                
                # MAPE (평균절대백분율오차)
                try:
                    evaluation["mape"] = mean_absolute_percentage_error(y_true, y_pred)
                except:
                    # sklearn 버전이 낮은 경우 직접 계산
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    evaluation["mape"] = mape
                
                # 잔차 분석
                residuals = y_true - y_pred
                evaluation["residual_analysis"] = {
                    "mean": float(np.mean(residuals)),
                    "std": float(np.std(residuals)),
                    "skewness": float(stats.skew(residuals)),
                    "kurtosis": float(stats.kurtosis(residuals)),
                    "normality_test": {
                        "shapiro_stat": float(stats.shapiro(residuals[:min(5000, len(residuals))])[0]),
                        "shapiro_pvalue": float(stats.shapiro(residuals[:min(5000, len(residuals))])[1])
                    }
                }
                
                # 예측 구간별 성능
                percentiles = [25, 50, 75]
                performance_by_range = {}
                for p in percentiles:
                    threshold = np.percentile(y_true, p)
                    mask = y_true <= threshold
                    if np.sum(mask) > 0:
                        range_r2 = r2_score(y_true[mask], y_pred[mask])
                        range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                        performance_by_range[f"bottom_{p}%"] = {
                            "r2": range_r2,
                            "mae": range_mae,
                            "samples": int(np.sum(mask))
                        }
                
                evaluation["performance_by_range"] = performance_by_range
            
            # 전체 통계
            evaluation["overall_stats"] = {
                "total_samples": len(y_true),
                "unique_predictions": len(np.unique(y_pred)),
                "unique_targets": len(np.unique(y_true))
            }
            
            # 성능 해석
            if task_type == "classification":
                acc = evaluation["accuracy"]
                if acc > 0.9:
                    performance_level = "매우 우수"
                elif acc > 0.8:
                    performance_level = "우수"
                elif acc > 0.7:
                    performance_level = "양호"
                elif acc > 0.6:
                    performance_level = "보통"
                else:
                    performance_level = "개선 필요"
                
                evaluation["interpretation"] = f"모델 정확도는 {acc:.3f}로 {performance_level}한 수준입니다."
            
            else:  # regression
                r2 = evaluation["r2"]
                if r2 > 0.9:
                    performance_level = "매우 높은 설명력"
                elif r2 > 0.7:
                    performance_level = "높은 설명력"
                elif r2 > 0.5:
                    performance_level = "중간 설명력"
                elif r2 > 0.3:
                    performance_level = "낮은 설명력"
                else:
                    performance_level = "매우 낮은 설명력"
                
                evaluation["interpretation"] = f"모델 R² 점수는 {r2:.3f}로 {performance_level}을 보입니다."
            
            return evaluation
            
        except Exception as e:
            return {"error": f"모델 평가 중 오류: {str(e)}"}
    
    @staticmethod
    def feature_selection(data: List[List[float]], 
                         target: List[Union[int, str, float]], 
                         feature_names: List[str] = None,
                         method: str = "auto",
                         k_features: int = None) -> Dict[str, Any]:
        """특성 선택"""
        try:
            from sklearn.feature_selection import (
                SelectKBest, f_classif, f_regression, 
                mutual_info_classif, mutual_info_regression,
                RFE, SelectFromModel
            )
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            
            # 데이터 변환
            X = np.array(data)
            y = np.array(target)
            
            if len(X) < 10:
                return {"error": "특성 선택을 위해서는 최소 10개 이상의 데이터가 필요합니다."}
            
            # 특성 이름 설정
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # k_features 기본값 설정
            if k_features is None:
                k_features = min(10, max(3, int(X.shape[1] * 0.7)))
            
            # 태스크 타입 추정
            task_type = "classification" if len(np.unique(y)) < 20 else "regression"
            
            # 타겟 인코딩 (분류에서 문자열인 경우)
            if task_type == "classification" and not np.issubdtype(y.dtype, np.number):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            selection_results = {}
            
            # 방법별 특성 선택
            if method == "auto" or method == "univariate":
                # 단변량 통계적 검정
                if task_type == "classification":
                    selector = SelectKBest(score_func=f_classif, k=k_features)
                else:
                    selector = SelectKBest(score_func=f_regression, k=k_features)
                
                X_selected = selector.fit_transform(X, y)
                selected_features = np.array(feature_names)[selector.get_support()].tolist()
                feature_scores = dict(zip(feature_names, selector.scores_))
                
                selection_results["univariate"] = {
                    "selected_features": selected_features,
                    "feature_scores": feature_scores,
                    "selected_count": len(selected_features)
                }
            
            if method == "auto" or method == "mutual_info":
                # 상호 정보량
                if task_type == "classification":
                    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
                else:
                    selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
                
                X_selected = selector.fit_transform(X, y)
                selected_features = np.array(feature_names)[selector.get_support()].tolist()
                feature_scores = dict(zip(feature_names, selector.scores_))
                
                selection_results["mutual_info"] = {
                    "selected_features": selected_features,
                    "feature_scores": feature_scores,
                    "selected_count": len(selected_features)
                }
            
            if method == "auto" or method == "rfe":
                # 재귀적 특성 제거
                if task_type == "classification":
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                
                selector = RFE(estimator, n_features_to_select=k_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = np.array(feature_names)[selector.get_support()].tolist()
                feature_ranking = dict(zip(feature_names, selector.ranking_))
                
                selection_results["rfe"] = {
                    "selected_features": selected_features,
                    "feature_ranking": feature_ranking,
                    "selected_count": len(selected_features)
                }
            
            if method == "auto" or method == "model_based":
                # 모델 기반 선택
                if task_type == "classification":
                    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
                estimator.fit(X, y)
                selector = SelectFromModel(estimator, max_features=k_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = np.array(feature_names)[selector.get_support()].tolist()
                feature_importance = dict(zip(feature_names, estimator.feature_importances_))
                
                selection_results["model_based"] = {
                    "selected_features": selected_features,
                    "feature_importance": feature_importance,
                    "selected_count": len(selected_features)
                }
            
            # 최종 추천 특성 (여러 방법의 교집합)
            if len(selection_results) > 1:
                all_selected = []
                for result in selection_results.values():
                    all_selected.extend(result["selected_features"])
                
                # 특성별 선택 횟수 계산
                feature_votes = {}
                for feature in all_selected:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
                
                # 가장 많이 선택된 특성들 추천
                recommended_features = sorted(
                    feature_votes.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:k_features]
                
                consensus_features = [feat for feat, votes in recommended_features if votes >= len(selection_results) // 2]
            else:
                consensus_features = list(selection_results.values())[0]["selected_features"]
            
            return {
                "task_type": task_type,
                "original_features": len(feature_names),
                "target_features": k_features,
                "selection_methods": list(selection_results.keys()),
                "results_by_method": selection_results,
                "consensus_features": consensus_features,
                "consensus_count": len(consensus_features),
                "feature_vote_summary": feature_votes if len(selection_results) > 1 else None,
                "interpretation": f"{len(selection_results)}개 방법을 사용하여 {len(feature_names)}개 특성 중 {len(consensus_features)}개의 핵심 특성을 선별했습니다."
            }
            
        except Exception as e:
            return {"error": f"특성 선택 중 오류: {str(e)}"}


# MCP 도구 등록
@mcp.tool("build_classification_model")
def build_classification_model(data: List[List[float]], 
                             target: List[Union[int, str]], 
                             feature_names: List[str] = None,
                             algorithm: str = "auto",
                             test_size: float = 0.2) -> Dict[str, Any]:
    """
    분류 모델을 구축하고 평가합니다.
    
    Args:
        data: 특성 데이터 (2차원 리스트)
        target: 타겟 클래스 (리스트)
        feature_names: 특성 이름들 (선택사항)
        algorithm: 알고리즘 선택 (auto, random_forest, gradient_boosting, svm, logistic, xgboost)
        test_size: 테스트 세트 비율
    
    Returns:
        분류 모델 구축 및 평가 결과
    """
    return AdvancedMLTools.build_classification_model(data, target, feature_names, algorithm, test_size)


@mcp.tool("build_regression_model")
def build_regression_model(data: List[List[float]], 
                         target: List[float], 
                         feature_names: List[str] = None,
                         algorithm: str = "auto",
                         test_size: float = 0.2) -> Dict[str, Any]:
    """
    회귀 모델을 구축하고 평가합니다.
    
    Args:
        data: 특성 데이터 (2차원 리스트)
        target: 타겟 값들 (리스트)
        feature_names: 특성 이름들 (선택사항)
        algorithm: 알고리즘 선택 (auto, random_forest, gradient_boosting, svr, linear, ridge, lasso, xgboost)
        test_size: 테스트 세트 비율
    
    Returns:
        회귀 모델 구축 및 평가 결과
    """
    return AdvancedMLTools.build_regression_model(data, target, feature_names, algorithm, test_size)


@mcp.tool("hyperparameter_tuning")
def hyperparameter_tuning(data: List[List[float]], 
                        target: List[Union[int, str, float]], 
                        model_type: str = "random_forest",
                        task_type: str = "classification",
                        cv_folds: int = 5) -> Dict[str, Any]:
    """
    하이퍼파라미터 튜닝을 수행합니다.
    
    Args:
        data: 특성 데이터
        target: 타겟 데이터
        model_type: 모델 타입 (random_forest, svm, logistic)
        task_type: 태스크 타입 (classification, regression)
        cv_folds: 교차 검증 폴드 수
    
    Returns:
        최적 하이퍼파라미터 및 성능 결과
    """
    return AdvancedMLTools.hyperparameter_tuning(data, target, model_type, task_type, cv_folds)


@mcp.tool("model_evaluation")
def model_evaluation(data: List[List[float]], 
                    target: List[Union[int, str, float]], 
                    predictions: List[Union[int, str, float]],
                    task_type: str = "classification") -> Dict[str, Any]:
    """
    모델 성능을 종합적으로 평가합니다.
    
    Args:
        data: 특성 데이터 (선택사항)
        target: 실제 타겟 값
        predictions: 모델 예측 값
        task_type: 태스크 타입 (classification, regression)
    
    Returns:
        상세한 모델 평가 결과
    """
    return AdvancedMLTools.model_evaluation(data, target, predictions, task_type)


@mcp.tool("feature_selection")
def feature_selection(data: List[List[float]], 
                     target: List[Union[int, str, float]], 
                     feature_names: List[str] = None,
                     method: str = "auto",
                     k_features: int = None) -> Dict[str, Any]:
    """
    특성 선택을 수행합니다.
    
    Args:
        data: 특성 데이터
        target: 타겟 데이터
        feature_names: 특성 이름들
        method: 선택 방법 (auto, univariate, mutual_info, rfe, model_based)
        k_features: 선택할 특성 수
    
    Returns:
        특성 선택 결과 및 중요도
    """
    return AdvancedMLTools.feature_selection(data, target, feature_names, method, k_features)


@mcp.tool("compare_models")
def compare_models(data: List[List[float]], 
                  target: List[Union[int, str, float]], 
                  algorithms: List[str] = None,
                  task_type: str = "auto",
                  metrics: List[str] = None) -> Dict[str, Any]:
    """
    여러 모델을 비교 평가합니다.
    
    Args:
        data: 특성 데이터
        target: 타겟 데이터
        algorithms: 비교할 알고리즘 리스트
        task_type: 태스크 타입 (auto, classification, regression)
        metrics: 평가 지표 리스트
    
    Returns:
        모델 비교 결과
    """
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
        
        # 데이터 변환
        X = np.array(data)
        y = np.array(target)
        
        # 태스크 타입 자동 추정
        if task_type == "auto":
            task_type = "classification" if len(np.unique(y)) < 20 else "regression"
        
        # 타겟 인코딩
        label_encoder = None
        if task_type == "classification" and not np.issubdtype(y.dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # 기본 알고리즘 설정
        if algorithms is None:
            if task_type == "classification":
                algorithms = ["random_forest", "gradient_boosting", "svm", "logistic"]
            else:
                algorithms = ["random_forest", "gradient_boosting", "svr", "linear"]
        
        # 각 알고리즘별 결과 수집
        comparison_results = {}
        
        for algorithm in algorithms:
            try:
                if task_type == "classification":
                    result = AdvancedMLTools.build_classification_model(
                        data, target.tolist() if hasattr(target, 'tolist') else target, 
                        algorithm=algorithm
                    )
                else:
                    result = AdvancedMLTools.build_regression_model(
                        data, target.tolist() if hasattr(target, 'tolist') else target, 
                        algorithm=algorithm
                    )
                
                if "error" not in result:
                    comparison_results[algorithm] = result
                    
            except Exception as e:
                comparison_results[algorithm] = {"error": str(e)}
        
        # 최고 모델 선택
        best_algorithm = None
        best_score = -np.inf if task_type == "regression" else 0
        
        for algorithm, result in comparison_results.items():
            if "error" not in result:
                if task_type == "classification":
                    score = result.get("best_accuracy", 0)
                else:
                    score = result.get("best_r2", -np.inf)
                
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
        
        # 비교 요약
        summary = {
            "task_type": task_type,
            "algorithms_tested": len(algorithms),
            "successful_algorithms": len([r for r in comparison_results.values() if "error" not in r]),
            "best_algorithm": best_algorithm,
            "best_score": best_score,
            "data_info": {
                "samples": len(X),
                "features": X.shape[1],
                "target_classes": len(np.unique(y)) if task_type == "classification" else None
            }
        }
        
        return {
            "summary": summary,
            "detailed_results": comparison_results,
            "interpretation": f"{len(algorithms)}개 알고리즘 중 {best_algorithm}이 {best_score:.4f} 점수로 최고 성능을 보였습니다."
        }
        
    except Exception as e:
        return {"error": f"모델 비교 중 오류: {str(e)}"}


if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Advanced ML Tools MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)