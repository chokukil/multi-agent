# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
MCP Tool: Statistical Analysis Tools
통계 분석에 자주 사용되는 기술통계·가설검정·상관·회귀·ANOVA 기능을 MCP 툴 형태로 제공합니다.
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
warnings.filterwarnings("ignore")

# FastMCP import
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# 서버 포트 설정
SERVER_PORT = int(os.getenv("SERVER_PORT", "8018"))

# FastMCP 서버 생성
mcp = FastMCP("Statistical Analysis Tools")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalysisTools:
    """통계 분석 도구 클래스"""

    # ------------------------------------------------------------------
    # 1. 기술 통계
    # ------------------------------------------------------------------
    @staticmethod
    def descriptive_statistics(
        data: List[List[Union[float, str]]],
        column_names: List[str] = None,
        target_columns: List[str] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """기술 통계량을 계산합니다.

        Args:
            data: 2차원 리스트 형태의 데이터.
            column_names: 데이터프레임 칼럼 이름 리스트.
            target_columns: 분석할 칼럼 이름 리스트. 지정하지 않으면 모든 칼럼을 사용합니다.
            confidence_level: 신뢰구간 수준 (기본값 0.95).

        Returns:
            기술 통계 분석 결과 딕셔너리.
        """
        try:
            # 입력 데이터프레임 생성
            df = pd.DataFrame(data, columns=column_names) if column_names else pd.DataFrame(data)

            # 분석 대상 칼럼 선택
            cols_to_analyze = (
                [c for c in target_columns if c in df.columns]
                if target_columns
                else df.columns.tolist()
            )

            results: Dict[str, Any] = {}

            for col in cols_to_analyze:
                col_data = df[col].dropna()

                # -------------------------------
                # 1‑1. 수치형 변수
                # -------------------------------
                if pd.api.types.is_numeric_dtype(col_data):
                    from scipy import stats

                    n = len(col_data)
                    mean = float(col_data.mean())
                    median = float(col_data.median())
                    std = float(col_data.std())
                    var = float(col_data.var())

                    # 사분위수 및 IQR
                    q1 = float(col_data.quantile(0.25))
                    q3 = float(col_data.quantile(0.75))
                    iqr = q3 - q1

                    # 왜도·첨도
                    skewness = float(col_data.skew())
                    kurtosis = float(col_data.kurtosis())

                    # 신뢰구간(평균)
                    confidence_interval = stats.t.interval(
                        confidence_level,
                        n - 1,
                        loc=mean,
                        scale=stats.sem(col_data),
                    )

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
                        "outliers": StatisticalAnalysisTools._detect_outliers_iqr(col_data),
                    }

                # -------------------------------
                # 1‑2. 범주형 변수
                # -------------------------------
                else:
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
                        "entropy": float(-np.sum(proportions * np.log2(proportions))),
                    }

            return {
                "descriptive_statistics": results,
                "summary": {
                    "total_variables": len(cols_to_analyze),
                    "numeric_variables": len([r for r in results.values() if r["type"] == "numeric"]),
                    "categorical_variables": len([r for r in results.values() if r["type"] == "categorical"]),
                    "confidence_level": confidence_level,
                },
                "interpretation": f"기술 통계 분석 완료: {len(cols_to_analyze)}개 변수 분석",
            }

        except Exception as e:
            return {"error": f"기술 통계 계산 오류: {str(e)}"}

    # ------------------------------------------------------------------
    # 1‑a. 이상치 탐지(IQR)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_outliers_iqr(data: pd.Series, threshold: float = 1.5) -> Dict[str, Any]:
        """IQR 방법으로 이상치를 탐지합니다."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(data)) * 100,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_values": outliers.tolist()[:10],  # 최대 10개만 표시
        }

    # ------------------------------------------------------------------
    # 2. 가설 검정
    # ------------------------------------------------------------------
    @staticmethod
    def hypothesis_testing(
        data1: List[float],
        data2: List[float] = None,
        test_type: str = "auto",
        alternative: str = "two-sided",
        alpha: float = 0.05,
        population_mean: float = 0.0,
    ) -> Dict[str, Any]:
        """가설 검정(one‑sample / two‑sample)을 수행합니다."""
        try:
            from scipy import stats

            sample1 = np.array(data1)
            sample1 = sample1[~np.isnan(sample1)]  # NaN 제거

            if len(sample1) < 3:
                return {"error": "가설 검정에는 최소 3개의 샘플 값이 필요합니다."}

            results: Dict[str, Any] = {}

            # ----------------------------------------------------------
            # 2‑1. 단일 표본 검정
            # ----------------------------------------------------------
            if data2 is None:
                results["test_category"] = "one_sample"

                # (i) 단일 표본 t‑검정
                t_stat, t_p = stats.ttest_1samp(sample1, population_mean)
                results["t_test"] = {
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "significant": t_p < alpha,
                    "null_hypothesis": f"모평균 = {population_mean}",
                    "alternative": alternative,
                }

                # (ii) Wilcoxon signed‑rank 검정 (표본 크기 ≥ 6 권장)
                if len(sample1) >= 6:
                    try:
                        w_stat, w_p = stats.wilcoxon(sample1 - population_mean)
                        results["wilcoxon_test"] = {
                            "statistic": float(w_stat),
                            "p_value": float(w_p),
                            "significant": w_p < alpha,
                        }
                    except Exception:
                        results["wilcoxon_test"] = {"error": "Wilcoxon 검정을 수행할 수 없습니다."}

                # (iii) 정규성 검정
                if len(sample1) >= 3:
                    s_stat, s_p = stats.shapiro(sample1)
                    results.setdefault("normality_test", {})["shapiro_wilk"] = {
                        "statistic": float(s_stat),
                        "p_value": float(s_p),
                        "is_normal": s_p > alpha,
                    }

            # ----------------------------------------------------------
            # 2‑2. 두 표본 검정
            # ----------------------------------------------------------
            else:
                sample2 = np.array(data2)
                sample2 = sample2[~np.isnan(sample2)]

                if len(sample2) < 3:
                    return {"error": "두 번째 표본에도 최소 3개의 샘플 값이 필요합니다."}

                results["test_category"] = "two_sample"

                # (i) 독립 표본 t‑검정 (등분산 가정)
                t_stat, t_p = stats.ttest_ind(sample1, sample2, equal_var=True)
                results["t_test_equal_var"] = {
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "significant": t_p < alpha,
                    "null_hypothesis": "두 집단의 평균이 같다.",
                }

                # (ii) Welch t‑검정 (등분산 가정 X)
                t_stat_w, t_p_w = stats.ttest_ind(sample1, sample2, equal_var=False)
                results["welch_t_test"] = {
                    "statistic": float(t_stat_w),
                    "p_value": float(t_p_w),
                    "significant": t_p_w < alpha,
                }

                # (iii) Mann‑Whitney U 검정 (비모수)
                u_stat, u_p = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
                results["mann_whitney_u"] = {
                    "statistic": float(u_stat),
                    "p_value": float(u_p),
                    "significant": u_p < alpha,
                }

                # (iv) 등분산 검정 – Levene
                lev_stat, lev_p = stats.levene(sample1, sample2)
                results["levene_test"] = {
                    "statistic": float(lev_stat),
                    "p_value": float(lev_p),
                    "equal_variances": lev_p > alpha,
                }

                # (v) 효과크기 – Cohen's d
                pooled_std = np.sqrt(
                    ((len(sample1) - 1) * np.var(sample1, ddof=1) + (len(sample2) - 1) * np.var(sample2, ddof=1))
                    / (len(sample1) + len(sample2) - 2)
                )
                cohens_d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
                results["effect_size"] = {
                    "cohens_d": float(cohens_d),
                    "interpretation": StatisticalAnalysisTools._interpret_effect_size(cohens_d),
                }

                # (vi) 각 표본 정규성 검정
                s1_stat, s1_p = stats.shapiro(sample1)
                s2_stat, s2_p = stats.shapiro(sample2)
                results["normality_tests"] = {
                    "sample1": {
                        "statistic": float(s1_stat),
                        "p_value": float(s1_p),
                        "is_normal": s1_p > alpha,
                    },
                    "sample2": {
                        "statistic": float(s2_stat),
                        "p_value": float(s2_p),
                        "is_normal": s2_p > alpha,
                    },
                }

            # 공통 표본 정보
            results["sample_info"] = {
                "sample1_size": len(sample1),
                "sample1_mean": float(np.mean(sample1)),
                "sample1_std": float(np.std(sample1, ddof=1)),
                "sample2_size": len(sample2) if data2 is not None else None,
                "sample2_mean": float(np.mean(sample2)) if data2 is not None else None,
                "sample2_std": float(np.std(sample2, ddof=1)) if data2 is not None else None,
                "alpha": alpha,
            }

            return results

        except Exception as e:
            return {"error": f"가설 검정 오류: {str(e)}"}

    # ------------------------------------------------------------------
    # 2‑a. 효과크기 해석
    # ------------------------------------------------------------------
    @staticmethod
    def _interpret_effect_size(effect_size: float) -> str:
        """Cohen's d 해석"""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "매우 약한 효과"
        elif abs_effect < 0.5:
            return "약한 효과"
        elif abs_effect < 0.8:
            return "중간 효과"
        else:
            return "큰 효과"

    # ------------------------------------------------------------------
    # 3. 상관 분석
    # ------------------------------------------------------------------
    @staticmethod
    def correlation_analysis(
        data: List[List[float]],
        column_names: List[str] = None,
        method: str = "pearson",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """상관 분석을 수행합니다 (Pearson / Spearman / Kendall)."""
        try:
            from scipy import stats

            df = pd.DataFrame(data, columns=column_names) if column_names else pd.DataFrame(data)
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            if numeric_df.shape[1] < 2:
                return {"error": "상관 분석을 위해서는 최소 2개의 수치형 변수가 필요합니다."}

            # 상관계수 행렬
            if method not in {"pearson", "spearman", "kendall"}:
                return {"error": f"지원하지 않는 상관 분석 방법: {method}"}
            corr_matrix = numeric_df.corr(method=method)
            results: Dict[str, Any] = {"correlation_matrix": corr_matrix.to_dict()}

            # 쌍별 상관계수 및 유의성
            correlations: Dict[str, Any] = {}
            vars_ = numeric_df.columns.tolist()
            for i, v1 in enumerate(vars_):
                for j, v2 in enumerate(vars_):
                    if i < j:
                        x, y = numeric_df[v1].values, numeric_df[v2].values
                        if method == "pearson":
                            coef, p = stats.pearsonr(x, y)
                        elif method == "spearman":
                            coef, p = stats.spearmanr(x, y)
                        else:
                            coef, p = stats.kendalltau(x, y)
                        correlations[f"{v1}_vs_{v2}"] = {
                            "correlation": float(coef),
                            "p_value": float(p),
                            "significant": p < alpha,
                            "strength": StatisticalAnalysisTools._interpret_correlation_strength(abs(coef)),
                        }
            results["pairwise_correlations"] = correlations

            # 강한 상관(>|0.7| & 유의)
            strong = {
                k: v for k, v in correlations.items() if abs(v["correlation"]) > 0.7 and v["significant"]
            }
            results["strong_correlations"] = strong

            all_corrs = [v["correlation"] for v in correlations.values()]
            results["summary"] = {
                "method": method,
                "total_pairs": len(correlations),
                "significant_pairs": len([v for v in correlations.values() if v["significant"]]),
                "strong_correlations": len(strong),
                "mean_correlation": float(np.mean(all_corrs)),
                "max_correlation": float(np.max(all_corrs)),
                "min_correlation": float(np.min(all_corrs)),
                "sample_size": len(numeric_df),
            }
            return results

        except Exception as e:
            return {"error": f"상관 분석 오류: {str(e)}"}

    # ------------------------------------------------------------------
    # 3‑a. 상관계수 강도 해석
    # ------------------------------------------------------------------
    @staticmethod
    def _interpret_correlation_strength(correlation: float) -> str:
        if correlation < 0.1:
            return "매우 약함"
        elif correlation < 0.3:
            return "약함"
        elif correlation < 0.5:
            return "보통"
        elif correlation < 0.7:
            return "강함"
        else:
            return "매우 강함"

    # ------------------------------------------------------------------
    # 4. 회귀 분석
    # ------------------------------------------------------------------
    @staticmethod
    def regression_analysis(
        data: List[List[float]],
        column_names: List[str] = None,
        dependent_var: str = None,
        independent_vars: List[str] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """다중 선형 회귀 분석을 수행합니다."""
        try:
            import statsmodels.api as sm
            from sklearn.metrics import r2_score, mean_squared_error
            from scipy import stats

            df = pd.DataFrame(data, columns=column_names) if column_names else pd.DataFrame(data)
            if column_names is None:
                df.columns = [f"var_{i}" for i in range(df.shape[1])]

            dependent_var = dependent_var or df.columns[0]
            independent_vars = independent_vars or [c for c in df.columns if c != dependent_var]

            # 변수 존재 및 수치형 검사
            for var in [dependent_var] + independent_vars:
                if var not in df.columns:
                    return {"error": f"변수 '{var}'가 데이터프레임에 존재하지 않습니다."}
                if not pd.api.types.is_numeric_dtype(df[var]):
                    return {"error": f"변수 '{var}'는 수치형이어야 합니다."}

            reg_df = df[[dependent_var] + independent_vars].dropna()
            if reg_df.shape[0] < len(independent_vars) + 2:
                return {"error": "회귀 분석을 수행하기에 충분한 데이터가 없습니다."}

            y = reg_df[dependent_var]
            X = sm.add_constant(reg_df[independent_vars])

            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)
            residuals = y - y_pred

            # 잔차 진단
            s_stat, s_p = stats.shapiro(residuals) if len(residuals) <= 5000 else (np.nan, np.nan)
            durbin_watson = sm.stats.durbin_watson(residuals)
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
            except Exception:
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
                    "sample_size": len(reg_df),
                },
                "coefficients": {
                    "variables": model.params.index.tolist(),
                    "estimates": model.params.values.tolist(),
                    "std_errors": model.bse.values.tolist(),
                    "t_values": model.tvalues.values.tolist(),
                    "p_values": model.pvalues.values.tolist(),
                    "confidence_intervals": model.conf_int(alpha=alpha).values.tolist(),
                    "significant_predictors": [
                        var for var, p in zip(model.params.index, model.pvalues) if p < alpha and var != "const"
                    ],
                },
                "residual_diagnostics": {
                    "normality_test": {
                        "statistic": float(s_stat) if not np.isnan(s_stat) else None,
                        "p_value": float(s_p) if not np.isnan(s_p) else None,
                        "is_normal": s_p > alpha if not np.isnan(s_p) else None,
                    },
                    "durbin_watson": float(durbin_watson),
                    "independence_assumption": 1.5 <= durbin_watson <= 2.5,
                    "homoscedasticity_test": {
                        "statistic": float(bp_stat) if not np.isnan(bp_stat) else None,
                        "p_value": float(bp_p) if not np.isnan(bp_p) else None,
                        "homoscedastic": bp_p > alpha if not np.isnan(bp_p) else None,
                    },
                },
                "variables": {
                    "dependent": dependent_var,
                    "independent": independent_vars,
                },
                "model_significance": model.f_pvalue < alpha,
                "interpretation": StatisticalAnalysisTools._interpret_regression_results(model, alpha),
            }
            return results

        except Exception as e:
            return {"error": f"회귀 분석 오류: {str(e)}"}

    # ------------------------------------------------------------------
    # 4‑a. 회귀 결과 해석
    # ------------------------------------------------------------------
    @staticmethod
    def _interpret_regression_results(model, alpha: float) -> str:
        """회귀 결과 요약 해석"""
        parts: List[str] = []

        # (1) 모델 유의성
        parts.append(
            "모델이 통계적으로 유의합니다." if model.f_pvalue < alpha else "모델이 통계적으로 유의하지 않습니다."
        )

        # (2) 설명력(R²)
        r2 = model.rsquared
        if r2 > 0.7:
            parts.append(f"높은 설명력을 보입니다 (R² = {r2:.3f}).")
        elif r2 > 0.5:
            parts.append(f"중간 수준의 설명력을 보입니다 (R² = {r2:.3f}).")
        else:
            parts.append(f"낮은 설명력을 보입니다 (R² = {r2:.3f}).")

        # (3) 유의한 독립 변수
        sig_vars = [v for v, p in zip(model.params.index, model.pvalues) if p < alpha and v != "const"]
        if sig_vars:
            parts.append(f"유의한 독립 변수: {', '.join(sig_vars)}")
        else:
            parts.append("유의한 독립 변수가 없습니다.")

        return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# MCP Tool 정의
# ──────────────────────────────────────────────────────────────

@mcp.tool()
def descriptive_statistics_tool(
    data: List[List[Union[float, str]]],
    column_names: List[str] = None,
    target_columns: List[str] = None,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """기술 통계량을 계산합니다."""
    return StatisticalAnalysisTools.descriptive_statistics(
        data, column_names, target_columns, confidence_level
    )


@mcp.tool()
def hypothesis_testing_tool(
    data1: List[float],
    data2: List[float] = None,
    test_type: str = "auto",
    alternative: str = "two-sided",
    alpha: float = 0.05,
    population_mean: float = 0.0,
) -> Dict[str, Any]:
    """가설 검정을 수행합니다."""
    return StatisticalAnalysisTools.hypothesis_testing(
        data1, data2, test_type, alternative, alpha, population_mean
    )


@mcp.tool()
def correlation_analysis_tool(
    data: List[List[float]],
    column_names: List[str] = None,
    method: str = "pearson",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """상관 분석을 수행합니다."""
    return StatisticalAnalysisTools.correlation_analysis(
        data, column_names, method, alpha
    )


@mcp.tool()
def regression_analysis_tool(
    data: List[List[float]],
    column_names: List[str] = None,
    dependent_var: str = None,
    independent_vars: List[str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """회귀 분석을 수행합니다."""
    return StatisticalAnalysisTools.regression_analysis(
        data, column_names, dependent_var, independent_vars, alpha
    )


# ──────────────────────────────────────────────────────────────
# 서버 실행
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Statistical Analysis Tools server...")
    logger.info(f"Server port: {SERVER_PORT}")
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=SERVER_PORT)
