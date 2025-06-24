# -*- coding: utf-8 -*-
"""
MCP Tool ▶ Semiconductor Yield Analysis
반도체 수율·품질 분석에 필요한 핵심 지표(웨이퍼 수율, 공정 수율, DPMO, Bin Map, 트렌드 예측)를 MCP 툴 형태로 제공합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import uvicorn

from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------
# 서버 설정
# ------------------------------------------------------------
SERVER_PORT = int(os.getenv("SERVER_PORT", "8008"))

mcp = FastMCP("Semiconductor Yield Analysis")

# ------------------------------------------------------------
# 핵심 로직 클래스
# ------------------------------------------------------------
class YieldAnalyzer:
    """반도체 수율·품질 분석 도구"""

    # --------------------------------------------------------
    # 1. 웨이퍼 수율
    # --------------------------------------------------------
    @staticmethod
    def calculate_wafer_yield(
        good_dies: int, total_dies: int, defect_density: float | None = None
    ) -> Dict[str, Any]:
        """웨이퍼 수율(%)을 계산합니다.

        Args:
            good_dies: 양품 다이 수.
            total_dies: 총 다이 수.
            defect_density: 결함 밀도(Defects / cm²) – 선택 입력.
        """
        if total_dies <= 0:
            return {"error": "total_dies 값은 0보다 커야 합니다."}
        if good_dies < 0 or good_dies > total_dies:
            return {"error": "good_dies 값이 유효 범위를 벗어났습니다."}

        yield_percent = good_dies / total_dies * 100
        result = {
            "wafer_yield_percent": round(yield_percent, 2),
            "good_dies": good_dies,
            "total_dies": total_dies,
            "defective_dies": total_dies - good_dies,
            "defect_rate_percent": round((total_dies - good_dies) / total_dies * 100, 2),
        }

        # 결함 밀도 기반 이론 수율 (Poisson 모델)
        if defect_density is not None and defect_density >= 0:
            die_area_cm2 = 1.0  # 기본값: 1 cm² (필요 시 파라미터화)
            theoretical_yield = math.exp(-defect_density * die_area_cm2) * 100
            result["theoretical_yield_percent"] = round(theoretical_yield, 2)
            result["yield_gap_percent"] = round(theoretical_yield - yield_percent, 2)

        return result

    # --------------------------------------------------------
    # 2. 공정 단계별 수율
    # --------------------------------------------------------
    @staticmethod
    def analyze_process_yield(step_yields: List[float]) -> Dict[str, Any]:
        """공정 단계별 수율을 분석하고 병목을 식별합니다."""
        if not step_yields:
            return {"error": "step_yields 리스트가 비어 있습니다."}
        if any(y < 0 or y > 100 for y in step_yields):
            return {"error": "각 단계 수율은 0~100 사이의 값이어야 합니다."}

        overall_yield = float(np.prod([y / 100 for y in step_yields]) * 100)
        bottleneck_idx = int(np.argmin(step_yields))
        improvement_scenarios: List[Dict[str, Any]] = []

        # 각 단계를 5 %p 향상시켰을 때 전체 수율 변화 시뮬레이션
        for i, y in enumerate(step_yields):
            improved = step_yields.copy()
            improved[i] = min(y + 5, 100)
            improved_overall = float(np.prod([v / 100 for v in improved]) * 100)
            improvement_scenarios.append(
                {
                    "step": i,
                    "current_yield": y,
                    "improved_yield": improved[i],
                    "overall_impact_percent": round(improved_overall - overall_yield, 2),
                }
            )

        return {
            "overall_yield_percent": round(overall_yield, 2),
            "step_yields_percent": step_yields,
            "bottleneck_step": bottleneck_idx,
            "bottleneck_yield_percent": step_yields[bottleneck_idx],
            "improvement_scenarios": improvement_scenarios,
            "total_steps": len(step_yields),
        }

    # --------------------------------------------------------
    # 3. DPMO & 시그마 수준
    # --------------------------------------------------------
    @staticmethod
    def calculate_dpmo(defects: int, units: int, opportunities: int) -> Dict[str, Any]:
        """DPMO(Defects Per Million Opportunities) 및 시그마 수준을 계산합니다."""
        if units <= 0 or opportunities <= 0:
            return {"error": "units/opportunities 값은 0보다 커야 합니다."}
        if defects < 0:
            return {"error": "defects 값은 음수가 될 수 없습니다."}

        dpmo = defects / (units * opportunities) * 1_000_000
        yield_rate = 1 - dpmo / 1_000_000  # 수율(0~1)

        # 간이 시그마 수준 추정 (Zbench ≈ NORMSINV(yield_rate) + 1.5)
        try:
            from scipy.stats import norm

            z_bench = norm.ppf(yield_rate)
            sigma_level = z_bench + 1.5
        except Exception:
            # SciPy 미설치 시 근사식 사용
            sigma_level = 6 if yield_rate >= 0.9999966 else max(0, (yield_rate - 0.3085) / 0.06)

        quality_level = (
            "Excellent" if sigma_level >= 5 else "Good" if sigma_level >= 4 else "Average" if sigma_level >= 3 else "Poor"
        )

        return {
            "dpmo": round(dpmo, 2),
            "sigma_level": round(sigma_level, 2),
            "yield_percent": round(yield_rate * 100, 4),
            "defects": defects,
            "units": units,
            "opportunities": opportunities,
            "quality_level": quality_level,
        }

    # --------------------------------------------------------
    # 4. Bin Map 분석
    # --------------------------------------------------------
    @staticmethod
    def analyze_bin_map(
        bin_data: List[List[int]], bin_descriptions: Dict[int, str] | None = None
    ) -> Dict[str, Any]:
        """2D Bin Map(패스/페일 등) 데이터를 분석합니다."""
        if not bin_data or not all(isinstance(row, list) for row in bin_data):
            return {"error": "bin_data 형식이 잘못되었습니다."}

        bin_array = np.asarray(bin_data)
        total_dies = int(bin_array.size)
        unique_bins, counts = np.unique(bin_array, return_counts=True)
        bin_counts = dict(zip(unique_bins.tolist(), counts.tolist()))

        # 기본 Bin 설명 (1: Pass, 0: Fail)
        descriptions = bin_descriptions or {1: "Pass", 0: "Fail"}
        bin_analysis: List[Dict[str, Any]] = []
        for bin_no, cnt in bin_counts.items():
            bin_analysis.append(
                {
                    "bin": int(bin_no),
                    "count": int(cnt),
                    "percentage": round(cnt / total_dies * 100, 2),
                    "description": descriptions.get(bin_no, f"Bin {bin_no}"),
                }
            )

        # 전체 수율(=Bin 1 비중)
        pass_count = bin_counts.get(1, 0)
        yield_percent = pass_count / total_dies * 100

        # 간단한 불량 클러스터 탐지(DFS)
        def _find_clusters(arr: np.ndarray, target: int) -> List[List[Tuple[int, int]]]:
            clusters: List[List[Tuple[int, int]]] = []
            visited = np.zeros_like(arr, dtype=bool)

            def dfs(r: int, c: int, cluster: List[Tuple[int, int]]):
                if (
                    r < 0
                    or r >= arr.shape[0]
                    or c < 0
                    or c >= arr.shape[1]
                    or visited[r, c]
                    or arr[r, c] != target
                ):
                    return
                visited[r, c] = True
                cluster.append((r, c))
                for dr, dc in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]:
                    dfs(r + dr, c + dc, cluster)

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if arr[i, j] == target and not visited[i, j]:
                        cluster: List[Tuple[int, int]] = []
                        dfs(i, j, cluster)
                        if cluster:
                            clusters.append(cluster)
            return clusters

        fail_clusters = _find_clusters(bin_array, 0)
        cluster_sizes = [len(c) for c in fail_clusters]
        cluster_analysis = {
            "total_clusters": len(fail_clusters),
            "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "cluster_sizes": cluster_sizes,
        }

        most_common_fail_bin = (
            max((b for b in bin_counts if b != 1), key=lambda b: bin_counts[b], default=None)
            if len(bin_counts) > 1
            else None
        )

        return {
            "total_dies": total_dies,
            "yield_percent": round(yield_percent, 2),
            "bin_analysis": sorted(bin_analysis, key=lambda x: x["count"], reverse=True),
            "cluster_analysis": cluster_analysis,
            "wafer_map_shape": f"{bin_array.shape[0]}x{bin_array.shape[1]}",
            "most_common_fail_bin": most_common_fail_bin,
        }

    # --------------------------------------------------------
    # 5. 수율 파레토 분석
    # --------------------------------------------------------
    @staticmethod
    def calculate_yield_pareto(defect_categories: Dict[str, int]) -> Dict[str, Any]:
        """파레토(80/20) 분석으로 주요 결함 항목을 파악합니다."""
        if not defect_categories:
            return {"error": "defect_categories 가 비어 있습니다."}

        sorted_items = sorted(defect_categories.items(), key=lambda x: x[1], reverse=True)
        total_defects = sum(defect_categories.values())
        pareto: List[Dict[str, Any]] = []
        cum_pct = 0
        for cat, cnt in sorted_items:
            pct = cnt / total_defects * 100
            cum_pct += pct
            pareto.append(
                {
                    "category": cat,
                    "count": cnt,
                    "percentage": round(pct, 2),
                    "cumulative_percentage": round(cum_pct, 2),
                    "is_vital_few": cum_pct <= 80,
                }
            )
        vital_few = [p for p in pareto if p["is_vital_few"]]
        vital_pct = sum(p["percentage"] for p in vital_few)

        return {
            "total_defects": total_defects,
            "pareto_analysis": pareto,
            "vital_few_count": len(vital_few),
            "vital_few_impact_percent": round(vital_pct, 2),
            "recommendation": f"상위 {len(vital_few)}개 항목이 전체 결함의 {round(vital_pct, 1)}%를 차지합니다.",
        }

    # --------------------------------------------------------
    # 6. 수율 트렌드 예측(선형 + 이동평균)
    # --------------------------------------------------------
    @staticmethod
    def predict_yield_trend(historical_yields: List[float], periods_ahead: int = 5) -> Dict[str, Any]:
        """과거 수율 데이터를 기반으로 단기 트렌드를 예측합니다."""
        if len(historical_yields) < 3:
            return {"error": "최소 3개의 과거 수율 데이터가 필요합니다."}
        if any(y < 0 or y > 100 for y in historical_yields):
            return {"error": "수율 데이터는 0~100 범위여야 합니다."}

        yields = np.asarray(historical_yields)
        n = len(yields)

        # 3 기간 이동평균
        ma3_last = np.mean(yields[-3:])

        # 단순 선형 추세
        x = np.arange(n)
        slope, intercept = np.polyfit(x, yields, 1)
        future_x = np.arange(n, n + periods_ahead)
        linear_pred = slope * future_x + intercept

        ma_pred = np.full(periods_ahead, ma3_last)

        trend = "Improving" if slope > 0.1 else "Declining" if slope < -0.1 else "Stable"
        volatility = float(np.std(yields))
        volatility_desc = "낮음" if volatility < 2 else "보통" if volatility < 5 else "높음"

        return {
            "historical_yields": historical_yields,
            "trend": trend,
            "slope": round(slope, 4),
            "volatility": round(volatility, 2),
            "current_ma3": round(ma3_last, 2),
            "linear_predictions": [round(p, 2) for p in linear_pred],
            "ma_predictions": [round(p, 2) for p in ma_pred],
            "confidence": "High" if volatility < 2 else "Medium" if volatility < 5 else "Low",
            "recommendation": f"현재 추세: {trend}, 변동성: {volatility_desc}",
        }

# ------------------------------------------------------------
# MCP 툴 래퍼 함수들
# ------------------------------------------------------------
@mcp.tool("calculate_wafer_yield")
def calculate_wafer_yield(
    good_dies: int, total_dies: int, defect_density: float | None = None
) -> Dict[str, Any]:
    """웨이퍼 수율 계산"""
    return YieldAnalyzer.calculate_wafer_yield(good_dies, total_dies, defect_density)


@mcp.tool("analyze_process_yield")
def analyze_process_yield(step_yields: List[float]) -> Dict[str, Any]:
    """공정 단계별 수율 분석"""
    return YieldAnalyzer.analyze_process_yield(step_yields)


@mcp.tool("calculate_dpmo")
def calculate_dpmo(defects: int, units: int, opportunities: int) -> Dict[str, Any]:
    """DPMO 및 시그마 수준 계산"""
    return YieldAnalyzer.calculate_dpmo(defects, units, opportunities)


@mcp.tool("analyze_bin_map")
def analyze_bin_map(bin_data: str, bin_descriptions: str | None = None) -> Dict[str, Any]:
    """Bin Map JSON 문자열을 분석"""
    try:
        bin_array = json.loads(bin_data)
        descriptions = json.loads(bin_descriptions) if bin_descriptions else None
        return YieldAnalyzer.analyze_bin_map(bin_array, descriptions)
    except json.JSONDecodeError:
        return {"error": "bin_data 또는 bin_descriptions 의 JSON 형식이 잘못되었습니다."}


@mcp.tool("calculate_yield_pareto")
def calculate_yield_pareto(defect_categories: Dict[str, int]) -> Dict[str, Any]:
    """수율 파레토 분석"""
    return YieldAnalyzer.calculate_yield_pareto(defect_categories)


@mcp.tool("predict_yield_trend")
def predict_yield_trend(historical_yields: List[float], periods_ahead: int = 5) -> Dict[str, Any]:
    """수율 트렌드 예측"""
    return YieldAnalyzer.predict_yield_trend(historical_yields, periods_ahead)


# ------------------------------------------------------------
# 실행 스크립트
# ------------------------------------------------------------
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Semiconductor Yield Analysis MCP server...")

    try:
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as exc:
        logger.error("Failed to start MCP server: %s", exc)
        raise
