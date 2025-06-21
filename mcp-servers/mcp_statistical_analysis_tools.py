#!/usr/bin/env python3
"""
MCP Tool: Statistical Analysis Tools
통계 분석 도구 - 기술통계, 추론통계, 가설검정, 상관분석, 회귀분석, 분산분석
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
SERVER_PORT = int(os.getenv('SERVER_PORT', '8018'))

# FastMCP 서버 생성
mcp = FastMCP("Statistical Analysis Tools")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalysisTools:
    """통계 분석 도구 클래스"""
    
    @staticmethod
    def descriptive_statistics(data: List[List[Union[float, str]]], 
                             column_names: List[str] = None,
                             target_columns: List[str] = None,
                             confidence_level: float = 0.95) -> Dict[str, Any]:
        """기술통계 계산"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            # 분석할 컬럼 선택
            if target_columns:
                cols_to_analyze = [col for col in target_columns if col in df.columns]
            else:
                cols_to_analyze = df.columns.tolist()
            
            results = {}
            
            for col in cols_to_analyze:
                col_data = df[col].dropna()
                
                if pd.api.types.is_numeric_dtype(col_data):
                    # 수치형 변수
                    from scipy import stats
                    
                    # 기본 통계량
                    n = len(col_data)
                    mean = float(col_data.mean())
                    median = float(col_data.median())
                    std = float(col_data.std())
                    var = float(col_data.var())
                    
                    # 분위수
                    q1 = float(col_data.quantile(0.25))
                    q3 = float(col_data.quantile(0.75))
                    iqr = q3 - q1
                    
                    # 형태 통계량
                    skewness = float(col_data.skew())
                    kurtosis = float(col_data.kurtosis())
                    
                    # 신뢰구간
                    confidence_interval = stats.t.interval(
                        confidence_level,
                        n - 1,
                        loc=mean,
                        scale=stats.sem(col_data)
                    )
                    
                    # 범위
                    range_val = float(col_data.max() - col_data.min())
                    cv = float(std / mean) if mean != 0 else np.inf
                    
                    results[col] = {
                        "type": "numeric",
                        "count": n,
                        "mean": mean,
                        "median": median,
                        "mode": float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                        "std": std,
                        "variance": var,
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "range": range_val,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "skewness": skewness,
                        "kurtosis": kurtosis,
                        "coefficient_of_variation": cv,
                        "confidence_interval": confidence_interval,
                        "outliers": self._detect_outliers_iqr(col_data)
                    }
                    
                else:
                    # 범주형 변수
                    value_counts = col_data.value_counts()
                    proportions = col_data.value_counts(normalize=True)
                    
                    results[col] = {
                        "type": "categorical",
                        "count": len(col_data),
                        "unique_values": len(value_counts),
                        "most_frequent": value_counts.index[0],
                        "mode_frequency": int(value_counts.iloc[0]),
                        "mode_proportion": float(proportions.iloc[0]),
                        "frequency_table": value_counts.to_dict(),
                        "proportion_table": proportions.to_dict(),
                        "entropy": float(-np.sum(proportions * np.log2(proportions)))
                    }
            
            return {
                "descriptive_statistics": results,
                "summary": {
                    "total_variables": len(cols_to_analyze),
                    "numeric_variables": len([r for r in results.values() if r["type"] == "numeric"]),
                    "categorical_variables": len([r for r in results.values() if r["type"] == "categorical"]),
                    "confidence_level": confidence_level
                },
                "interpretation": f"기술통계 분석 완료: {len(cols_to_analyze)}개 변수 분석"
            }
            
        except Exception as e:
            return {"error": f"기술통계 계산 중 오류: {str(e)}"}
    
    @staticmethod
    def _detect_outliers_iqr(data: pd.Series, threshold: float = 1.5) -> Dict[str, Any]:
        """IQR 방법으로 이상치 탐지"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(data)) * 100,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_values": outliers.tolist()[:10]  # 최대 10개만 표시
        }
    
    @staticmethod
    def hypothesis_testing(data1: List[float], 
                         data2: List[float] = None,
                         test_type: str = "auto",
                         alternative: str = "two-sided",
                         alpha: float = 0.05,
                         population_mean: float = 0.0) -> Dict[str, Any]:
        """가설검정 수행"""
        try:
            from scipy import stats
            
            # 데이터 변환
            sample1 = np.array(data1)
            sample1 = sample1[~np.isnan(sample1)]  # NaN 제거
            
            if len(sample1) < 3:
                return {"error": "가설검정을 위해서는 최소 3개 이상의 관측치가 필요합니다."}
            
            results = {}
            
            if data2 is None:
                # 일표본 검정
                results["test_category"] = "one_sample"
                
                # 일표본 t-검정
                t_stat, t_p = stats.ttest_1samp(sample1, population_mean)
                results["t_test"] = {
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "significant": t_p < alpha,
                    "null_hypothesis": f"평균 = {population_mean}",
                    "alternative": alternative
                }
                
                # 일표본 Wilcoxon 부호 순위 검정
                if len(sample1) >= 6:  # Wilcoxon 검정을 위한 최소 표본 크기
                    try:
                        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(sample1 - population_mean)
                        results["wilcoxon_test"] = {
                            "statistic": float(wilcoxon_stat),
                            "p_value": float(wilcoxon_p),
                            "significant": wilcoxon_p < alpha
                        }
                    except:
                        results["wilcoxon_test"] = {"error": "Wilcoxon 검정 수행 불가"}
                
                # 정규성 검정
                if len(sample1) >= 3:
                    shapiro_stat, shapiro_p = stats.shapiro(sample1)
                    results["normality_test"] = {
                        "shapiro_wilk": {
                            "statistic": float(shapiro_stat),
                            "p_value": float(shapiro_p),
                            "is_normal": shapiro_p > alpha
                        }
                    }
            
            else:
                # 이표본 검정
                sample2 = np.array(data2)
                sample2 = sample2[~np.isnan(sample2)]  # NaN 제거
                
                if len(sample2) < 3:
                    return {"error": "두 번째 표본도 최소 3개 이상의 관측치가 필요합니다."}
                
                results["test_category"] = "two_sample"
                
                # 독립표본 t-검정
                t_stat, t_p = stats.ttest_ind(sample1, sample2, equal_var=True)
                results["t_test_equal_var"] = {
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "significant": t_p < alpha,
                    "null_hypothesis": "두 집단의 평균이 같다"
                }
                
                # Welch's t-검정 (등분산 가정 없음)
                t_stat_welch, t_p_welch = stats.ttest_ind(sample1, sample2, equal_var=False)
                results["welch_t_test"] = {
                    "statistic": float(t_stat_welch),
                    "p_value": float(t_p_welch),
                    "significant": t_p_welch < alpha
                }
                
                # Mann-Whitney U 검정
                u_stat, u_p = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
                results["mann_whitney_u"] = {
                    "statistic": float(u_stat),
                    "p_value": float(u_p),
                    "significant": u_p < alpha
                }
                
                # Levene의 등분산 검정
                levene_stat, levene_p = stats.levene(sample1, sample2)
                results["levene_test"] = {
                    "statistic": float(levene_stat),
                    "p_value": float(levene_p),
                    "equal_variances": levene_p > alpha
                }
                
                # 효과 크기 (Cohen's d)
                pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1, ddof=1) + 
                                    (len(sample2) - 1) * np.var(sample2, ddof=1)) / 
                                   (len(sample1) + len(sample2) - 2))
                cohens_d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
                
                results["effect_size"] = {
                    "cohens_d": float(cohens_d),
                    "interpretation": StatisticalAnalysisTools._interpret_effect_size(cohens_d)
                }
                
                # 각 표본의 정규성 검정
                shapiro1_stat, shapiro1_p = stats.shapiro(sample1)
                shapiro2_stat, shapiro2_p = stats.shapiro(sample2)
                
                results["normality_tests"] = {
                    "sample1": {
                        "statistic": float(shapiro1_stat),
                        "p_value": float(shapiro1_p),
                        "is_normal": shapiro1_p > alpha
                    },
                    "sample2": {
                        "statistic": float(shapiro2_stat),
                        "p_value": float(shapiro2_p),
                        "is_normal": shapiro2_p > alpha
                    }
                }
            
            # 표본 정보
            results["sample_info"] = {
                "sample1_size": len(sample1),
                "sample1_mean": float(np.mean(sample1)),
                "sample1_std": float(np.std(sample1, ddof=1)),
                "sample2_size": len(sample2) if data2 is not None else None,
                "sample2_mean": float(np.mean(sample2)) if data2 is not None else None,
                "sample2_std": float(np.std(sample2, ddof=1)) if data2 is not None else None,
                "alpha": alpha
            }
            
            return results
            
        except Exception as e:
            return {"error": f"가설검정 중 오류: {str(e)}"}
    
    @staticmethod
    def _interpret_effect_size(effect_size: float) -> str:
        """효과 크기 해석"""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "작은 효과"
        elif abs_effect < 0.5:
            return "중간 효과"
        elif abs_effect < 0.8:
            return "큰 효과"
        else:
            return "매우 큰 효과"
    
    @staticmethod
    def correlation_analysis(data: List[List[float]], 
                           column_names: List[str] = None,
                           method: str = "pearson",
                           alpha: float = 0.05) -> Dict[str, Any]:
        """상관분석 수행"""
        try:
            from scipy import stats
            
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            # 수치형 컬럼만 선택
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_df.columns) < 2:
                return {"error": "상관분석을 위해서는 최소 2개의 수치형 변수가 필요합니다."}
            
            results = {}
            
            # 상관계수 행렬
            if method == "pearson":
                corr_matrix = numeric_df.corr(method='pearson')
            elif method == "spearman":
                corr_matrix = numeric_df.corr(method='spearman')
            elif method == "kendall":
                corr_matrix = numeric_df.corr(method='kendall')
            else:
                return {"error": f"지원하지 않는 상관분석 방법: {method}"}
            
            results["correlation_matrix"] = corr_matrix.to_dict()
            
            # 쌍별 상관분석 및 유의성 검정
            correlations = {}
            variables = numeric_df.columns.tolist()
            
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i < j:  # 상삼각형만 계산
                        data1 = numeric_df[var1].values
                        data2 = numeric_df[var2].values
                        
                        if method == "pearson":
                            corr_coef, p_value = stats.pearsonr(data1, data2)
                        elif method == "spearman":
                            corr_coef, p_value = stats.spearmanr(data1, data2)
                        elif method == "kendall":
                            corr_coef, p_value = stats.kendalltau(data1, data2)
                        
                        correlations[f"{var1}_vs_{var2}"] = {
                            "correlation": float(corr_coef),
                            "p_value": float(p_value),
                            "significant": p_value < alpha,
                            "strength": StatisticalAnalysisTools._interpret_correlation_strength(abs(corr_coef))
                        }
            
            results["pairwise_correlations"] = correlations
            
            # 강한 상관관계 식별
            strong_correlations = {
                pair: stats for pair, stats in correlations.items() 
                if abs(stats["correlation"]) > 0.7 and stats["significant"]
            }
            
            results["strong_correlations"] = strong_correlations
            
            # 요약 통계
            all_correlations = [stats["correlation"] for stats in correlations.values()]
            results["summary"] = {
                "method": method,
                "total_pairs": len(correlations),
                "significant_pairs": len([c for c in correlations.values() if c["significant"]]),
                "strong_correlations": len(strong_correlations),
                "mean_correlation": float(np.mean(all_correlations)),
                "max_correlation": float(np.max(all_correlations)),
                "min_correlation": float(np.min(all_correlations)),
                "sample_size": len(numeric_df)
            }
            
            return results
            
        except Exception as e:
            return {"error": f"상관분석 중 오류: {str(e)}"}
    
    @staticmethod
    def _interpret_correlation_strength(correlation: float) -> str:
        """상관계수 강도 해석"""
        if correlation < 0.1:
            return "매우 약함"
        elif correlation < 0.3:
            return "약함"
        elif correlation < 0.5:
            return "중간"
        elif correlation < 0.7:
            return "강함"
        else:
            return "매우 강함"
    
    @staticmethod
    def regression_analysis(data: List[List[float]], 
                          column_names: List[str] = None,
                          dependent_var: str = None,
                          independent_vars: List[str] = None,
                          alpha: float = 0.05) -> Dict[str, Any]:
        """회귀분석 수행"""
        try:
            import statsmodels.api as sm
            from sklearn.metrics import r2_score, mean_squared_error
            
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
                column_names = [f"var_{i}" for i in range(len(data[0]))]
                df.columns = column_names
            
            # 변수 설정
            if dependent_var is None:
                dependent_var = df.columns[0]
            
            if independent_vars is None:
                independent_vars = [col for col in df.columns if col != dependent_var]
            
            # 수치형 데이터만 사용
            all_vars = [dependent_var] + independent_vars
            for var in all_vars:
                if var not in df.columns:
                    return {"error": f"변수 '{var}'가 데이터에 존재하지 않습니다."}
                if not pd.api.types.is_numeric_dtype(df[var]):
                    return {"error": f"변수 '{var}'는 수치형이 아닙니다."}
            
            # 결측치 제거
            regression_data = df[all_vars].dropna()
            
            if len(regression_data) < len(independent_vars) + 2:
                return {"error": "회귀분석을 위한 충분한 데이터가 없습니다."}
            
            y = regression_data[dependent_var]
            X = regression_data[independent_vars]
            
            # 상수항 추가
            X = sm.add_constant(X)
            
            # OLS 회귀모델 적합
            model = sm.OLS(y, X).fit()
            
            # 예측값과 잔차
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            # 잔차 분석
            from scipy import stats
            
            # 정규성 검정 (잔차)
            shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) <= 5000 else (np.nan, np.nan)
            
            # Durbin-Watson 통계량
            durbin_watson = sm.stats.durbin_watson(residuals)
            
            # Breusch-Pagan 검정 (등분산성)
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_stat, bp_p, bp_f_stat, bp_f_p = het_breuschpagan(residuals, X)
            except:
                bp_stat, bp_p = np.nan, np.nan
            
            results = {
                "model_summary": {
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "f_statistic": float(model.fvalue),
                    "f_pvalue": float(model.f_pvalue),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "mse": float(mean_squared_error(y, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                    "sample_size": len(regression_data)
                },
                "coefficients": {
                    "variables": model.params.index.tolist(),
                    "estimates": model.params.values.tolist(),
                    "std_errors": model.bse.values.tolist(),
                    "t_values": model.tvalues.values.tolist(),
                    "p_values": model.pvalues.values.tolist(),
                    "confidence_intervals": model.conf_int(alpha=alpha).values.tolist(),
                    "significant_predictors": [
                        var for var, p in zip(model.params.index, model.pvalues) 
                        if p < alpha and var != "const"
                    ]
                },
                "residual_diagnostics": {
                    "normality_test": {
                        "statistic": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
                        "p_value": float(shapiro_p) if not np.isnan(shapiro_p) else None,
                        "is_normal": shapiro_p > alpha if not np.isnan(shapiro_p) else None
                    },
                    "durbin_watson": float(durbin_watson),
                    "independence_assumption": 1.5 <= durbin_watson <= 2.5,
                    "homoscedasticity_test": {
                        "statistic": float(bp_stat) if not np.isnan(bp_stat) else None,
                        "p_value": float(bp_p) if not np.isnan(bp_p) else None,
                        "homoscedastic": bp_p > alpha if not np.isnan(bp_p) else None
                    }
                },
                "variables": {
                    "dependent": dependent_var,
                    "independent": independent_vars
                },
                "model_significance": model.f_pvalue < alpha,
                "interpretation": StatisticalAnalysisTools._interpret_regression_results(model, alpha)
            }
            
            return results
            
        except Exception as e:
            return {"error": f"회귀분석 중 오류: {str(e)}"}
    
    @staticmethod
    def _interpret_regression_results(model, alpha: float) -> str:
        """회귀분석 결과 해석"""
        interpretation = []
        
        # 모델 유의성
        if model.f_pvalue < alpha:
            interpretation.append("모델이 통계적으로 유의합니다.")
        else:
            interpretation.append("모델이 통계적으로 유의하지 않습니다.")
        
        # 설명력
        r_squared = model.rsquared
        if r_squared > 0.7:
            interpretation.append(f"높은 설명력을 보입니다 (R² = {r_squared:.3f}).")
        elif r_squared > 0.5:
            interpretation.append(f"중간 정도의 설명력을 보입니다 (R² = {r_squared:.3f}).")
        else:
            interpretation.append(f"낮은 설명력을 보입니다 (R² = {r_squared:.3f}).")
        
        # 유의한 예측변수
        significant_vars = [var for var, p in zip(model.params.index, model.pvalues) 
                          if p < alpha and var != "const"]
        
        if significant_vars:
            interpretation.append(f"유의한 예측변수: {', '.join(significant_vars)}")
        else:
            interpretation.append("유의한 예측변수가 없습니다.")
        
        return " ".join(interpretation)
    
    @staticmethod
    def anova_analysis(data: List[List[Union[float, str]]], 
                     column_names: List[str] = None,
                     dependent_var: str = None,
                     factor_var: str = None,
                     alpha: float = 0.05) -> Dict[str, Any]:
        """분산분석 (ANOVA) 수행"""
        try:
            from scipy import stats
            
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
                column_names = [f"var_{i}" for i in range(len(data[0]))]
                df.columns = column_names
            
            # 변수 설정
            if dependent_var is None:
                dependent_var = df.select_dtypes(include=[np.number]).columns[0]
            
            if factor_var is None:
                factor_var = df.select_dtypes(include=['object', 'category']).columns[0]
            
            # 변수 검증
            if dependent_var not in df.columns:
                return {"error": f"종속변수 '{dependent_var}'가 존재하지 않습니다."}
            
            if factor_var not in df.columns:
                return {"error": f"요인변수 '{factor_var}'가 존재하지 않습니다."}
            
            if not pd.api.types.is_numeric_dtype(df[dependent_var]):
                return {"error": "종속변수는 수치형이어야 합니다."}
            
            # 결측치 제거
            anova_data = df[[dependent_var, factor_var]].dropna()
            
            # 그룹별 데이터 준비
            groups = [group[dependent_var].values for name, group in anova_data.groupby(factor_var)]
            group_names = [str(name) for name, group in anova_data.groupby(factor_var)]
            
            if len(groups) < 2:
                return {"error": "ANOVA 분석을 위해서는 최소 2개의 그룹이 필요합니다."}
            
            # 각 그룹의 최소 크기 확인
            min_group_size = min(len(group) for group in groups)
            if min_group_size < 2:
                return {"error": "모든 그룹은 최소 2개의 관측치가 필요합니다."}
            
            # 일원분산분석
            f_stat, f_p = stats.f_oneway(*groups)
            
            # Kruskal-Wallis 검정 (비모수 대안)
            kw_stat, kw_p = stats.kruskal(*groups)
            
            # Levene의 등분산 검정
            levene_stat, levene_p = stats.levene(*groups)
            
            # 그룹별 기술통계
            group_stats = {}
            for name, group in anova_data.groupby(factor_var):
                values = group[dependent_var]
                group_stats[str(name)] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median())
                }
            
            # 효과 크기 (eta squared)
            overall_mean = anova_data[dependent_var].mean()
            ss_between = sum([len(group) * (np.mean(group) - overall_mean)**2 for group in groups])
            ss_total = np.sum((anova_data[dependent_var] - overall_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # 사후검정 (Tukey HSD)
            post_hoc_results = {}
            if f_p < alpha and len(groups) > 2:
                try:
                    from scipy.stats import tukey_hsd
                    tukey_result = tukey_hsd(*groups)
                    
                    # 쌍별 비교 결과
                    pairwise_comparisons = {}
                    for i, group1 in enumerate(group_names):
                        for j, group2 in enumerate(group_names):
                            if i < j:
                                comparison_key = f"{group1}_vs_{group2}"
                                pairwise_comparisons[comparison_key] = {
                                    "mean_diff": float(np.mean(groups[i]) - np.mean(groups[j])),
                                    "significant": bool(tukey_result.pvalue[i, j] < alpha) if hasattr(tukey_result, 'pvalue') else None
                                }
                    
                    post_hoc_results = {
                        "method": "Tukey HSD",
                        "pairwise_comparisons": pairwise_comparisons
                    }
                except:
                    post_hoc_results = {"error": "사후검정 수행 불가"}
            
            results = {
                "anova_results": {
                    "f_statistic": float(f_stat),
                    "p_value": float(f_p),
                    "significant": f_p < alpha,
                    "eta_squared": float(eta_squared),
                    "effect_size_interpretation": StatisticalAnalysisTools._interpret_eta_squared(eta_squared)
                },
                "kruskal_wallis": {
                    "statistic": float(kw_stat),
                    "p_value": float(kw_p),
                    "significant": kw_p < alpha
                },
                "levene_test": {
                    "statistic": float(levene_stat),
                    "p_value": float(levene_p),
                    "equal_variances": levene_p > alpha
                },
                "group_statistics": group_stats,
                "post_hoc_tests": post_hoc_results,
                "sample_info": {
                    "total_sample_size": len(anova_data),
                    "number_of_groups": len(groups),
                    "group_names": group_names,
                    "dependent_variable": dependent_var,
                    "factor_variable": factor_var
                },
                "assumptions": {
                    "normality": "각 그룹의 정규성을 별도로 확인하세요",
                    "homogeneity_of_variance": levene_p > alpha,
                    "independence": "연구 설계에 따라 확인하세요"
                }
            }
            
            return results
            
        except Exception as e:
            return {"error": f"ANOVA 분석 중 오류: {str(e)}"}
    
    @staticmethod
    def _interpret_eta_squared(eta_squared: float) -> str:
        """Eta squared 효과 크기 해석"""
        if eta_squared < 0.01:
            return "작은 효과"
        elif eta_squared < 0.06:
            return "중간 효과"
        elif eta_squared < 0.14:
            return "큰 효과"
        else:
            return "매우 큰 효과"


# MCP 도구 등록
@mcp.tool
def descriptive_statistics(data: List[List[Union[float, str]]], 
                         column_names: List[str] = None,
                         target_columns: List[str] = None,
                         confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    기술통계를 계산합니다.
    
    Args:
        data: 분석할 데이터 (2차원 리스트)
        column_names: 컬럼 이름들
        target_columns: 분석할 특정 컬럼들
        confidence_level: 신뢰수준 (기본값: 0.95)
    
    Returns:
        기술통계 결과
    """
    return StatisticalAnalysisTools.descriptive_statistics(data, column_names, target_columns, confidence_level)


@mcp.tool
def hypothesis_testing(data1: List[float], 
                     data2: List[float] = None,
                     test_type: str = "auto",
                     alternative: str = "two-sided",
                     alpha: float = 0.05,
                     population_mean: float = 0.0) -> Dict[str, Any]:
    """
    가설검정을 수행합니다.
    
    Args:
        data1: 첫 번째 표본 데이터
        data2: 두 번째 표본 데이터 (선택사항)
        test_type: 검정 유형 (auto, t_test, wilcoxon, mann_whitney)
        alternative: 대립가설 (two-sided, less, greater)
        alpha: 유의수준 (기본값: 0.05)
        population_mean: 모평균 (일표본 검정용)
    
    Returns:
        가설검정 결과
    """
    return StatisticalAnalysisTools.hypothesis_testing(data1, data2, test_type, alternative, alpha, population_mean)


@mcp.tool
def correlation_analysis(data: List[List[float]], 
                       column_names: List[str] = None,
                       method: str = "pearson",
                       alpha: float = 0.05) -> Dict[str, Any]:
    """
    상관분석을 수행합니다.
    
    Args:
        data: 분석할 수치 데이터
        column_names: 컬럼 이름들
        method: 상관분석 방법 (pearson, spearman, kendall)
        alpha: 유의수준 (기본값: 0.05)
    
    Returns:
        상관분석 결과
    """
    return StatisticalAnalysisTools.correlation_analysis(data, column_names, method, alpha)


@mcp.tool
def regression_analysis(data: List[List[float]], 
                      column_names: List[str] = None,
                      dependent_var: str = None,
                      independent_vars: List[str] = None,
                      alpha: float = 0.05) -> Dict[str, Any]:
    """
    회귀분석을 수행합니다.
    
    Args:
        data: 분석할 수치 데이터
        column_names: 컬럼 이름들
        dependent_var: 종속변수명
        independent_vars: 독립변수명들
        alpha: 유의수준 (기본값: 0.05)
    
    Returns:
        회귀분석 결과
    """
    return StatisticalAnalysisTools.regression_analysis(data, column_names, dependent_var, independent_vars, alpha)


@mcp.tool
def anova_analysis(data: List[List[Union[float, str]]], 
                 column_names: List[str] = None,
                 dependent_var: str = None,
                 factor_var: str = None,
                 alpha: float = 0.05) -> Dict[str, Any]:
    """
    분산분석(ANOVA)을 수행합니다.
    
    Args:
        data: 분석할 데이터
        column_names: 컬럼 이름들
        dependent_var: 종속변수명 (수치형)
        factor_var: 요인변수명 (범주형)
        alpha: 유의수준 (기본값: 0.05)
    
    Returns:
        ANOVA 분석 결과
    """
    return StatisticalAnalysisTools.anova_analysis(data, column_names, dependent_var, factor_var, alpha)


@mcp.tool
def comprehensive_statistical_analysis(data: List[List[Union[float, str]]], 
                                     column_names: List[str] = None,
                                     analysis_types: List[str] = None,
                                     target_columns: List[str] = None,
                                     alpha: float = 0.05) -> Dict[str, Any]:
    """
    종합적인 통계 분석을 수행합니다.
    
    Args:
        data: 분석할 데이터
        column_names: 컬럼 이름들
        analysis_types: 수행할 분석 유형 리스트 (descriptive, correlation, normality)
        target_columns: 분석할 특정 컬럼들
        alpha: 유의수준 (기본값: 0.05)
    
    Returns:
        종합 통계 분석 결과
    """
    try:
        if analysis_types is None:
            analysis_types = ["descriptive", "correlation", "normality"]
        
        # 데이터 변환
        if column_names:
            df = pd.DataFrame(data, columns=column_names)
        else:
            df = pd.DataFrame(data)
        
        # 분석할 컬럼 선택
        if target_columns:
            cols_to_analyze = [col for col in target_columns if col in df.columns]
        else:
            cols_to_analyze = df.columns.tolist()
        
        results = {"comprehensive_analysis": {}}
        
        # 기술통계
        if "descriptive" in analysis_types:
            desc_result = StatisticalAnalysisTools.descriptive_statistics(
                data, column_names, cols_to_analyze
            )
            results["comprehensive_analysis"]["descriptive_statistics"] = desc_result
        
        # 상관분석 (수치형 변수가 2개 이상인 경우)
        if "correlation" in analysis_types:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                numeric_data = df[numeric_cols].values.tolist()
                corr_result = StatisticalAnalysisTools.correlation_analysis(
                    numeric_data, numeric_cols
                )
                results["comprehensive_analysis"]["correlation_analysis"] = corr_result
        
        # 정규성 검정
        if "normality" in analysis_types:
            from scipy import stats
            normality_results = {}
            
            for col in cols_to_analyze:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    col_data = df[col].dropna()
                    if len(col_data) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(col_data) if len(col_data) <= 5000 else (np.nan, np.nan)
                        normality_results[col] = {
                            "shapiro_wilk": {
                                "statistic": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
                                "p_value": float(shapiro_p) if not np.isnan(shapiro_p) else None,
                                "is_normal": shapiro_p > alpha if not np.isnan(shapiro_p) else None
                            }
                        }
            
            if normality_results:
                results["comprehensive_analysis"]["normality_tests"] = normality_results
        
        # 요약 정보
        results["analysis_summary"] = {
            "total_variables": len(cols_to_analyze),
            "numeric_variables": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_variables": len(df.select_dtypes(include=['object', 'category']).columns),
            "sample_size": len(df),
            "analyses_performed": analysis_types,
            "significance_level": alpha
        }
        
        return results
        
    except Exception as e:
        return {"error": f"종합 통계 분석 중 오류: {str(e)}"}


if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Statistical Analysis Tools MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)