# -*- coding: utf-8 -*-
"""
MCP Tool ▶ Process Control Charts
SPC 관리도, 공정 Capability(Cp/Cpk), 패턴 탐지, 게이지 R&R을 지원합니다.
깨진 한글·문법 오류를 복구하고 입력 검증·예외 처리를 강화했습니다.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import uvicorn
from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------
# 서버 설정
# ------------------------------------------------------------
SERVER_PORT = int(os.getenv("SERVER_PORT", "8009"))

mcp = FastMCP("Process Control Charts")

# ------------------------------------------------------------
# 분석 클래스
# ------------------------------------------------------------
class ProcessControlAnalyzer:
    """관리도·Capability·SPC 패턴·Gauge R&R 유틸리티"""

    # -----------------------------------------------------
    # 1. 관리한계 계산
    # -----------------------------------------------------
    @staticmethod
    def calculate_control_limits(data: List[float], chart_type: str = "xbar") -> Dict[str, Any]:
        if not data or len(data) < 2:
            return {"error": "데이터 포인트가 부족합니다."}
        arr = np.asarray(data, dtype=float)
        n = len(arr)
        chart_type = chart_type.lower()

        if chart_type == "xbar":
            center = float(arr.mean())
            moving_ranges = np.abs(np.diff(arr))
            avg_mr = float(moving_ranges.mean()) if moving_ranges.size else 0.0
            d2 = 1.128  # n=2 일 때
            sigma_est = avg_mr / d2 if d2 else 0
            ucl = center + 3 * sigma_est
            lcl = center - 3 * sigma_est
            warn_ucl = center + 2 * sigma_est
            warn_lcl = center - 2 * sigma_est
            variation = sigma_est

        elif chart_type == "r":
            ranges = np.abs(np.diff(arr))
            if ranges.size == 0:
                return {"error": "R 관리도에는 최소 2개의 포인트가 필요합니다."}
            center = float(ranges.mean())
            D3, D4 = 0.0, 3.267  # n=2
            ucl = D4 * center
            lcl = D3 * center
            warn_ucl = center + 2 / 3 * (ucl - center)
            warn_lcl = max(0.0, center - 2 / 3 * (center - lcl))
            variation = float(ranges.std(ddof=1))
        else:
            return {"error": f"지원되지 않는 관리도 유형: {chart_type}"}

        # 이상점 탐지
        ooc = [
            {"index": i, "value": float(v), "type": "beyond_limits"}
            for i, v in enumerate(arr)
            if v > ucl or v < lcl
        ]

        return {
            "chart_type": chart_type,
            "centerline": round(center, 4),
            "ucl": round(ucl, 4),
            "lcl": round(lcl, 4),
            "warning_ucl": round(warn_ucl, 4),
            "warning_lcl": round(warn_lcl, 4),
            "out_of_control_points": ooc,
            "process_in_control": not ooc,
            "data_points": n,
            "process_variation_est": round(variation, 4),
        }

    # -----------------------------------------------------
    # 2. Cp / Cpk
    # -----------------------------------------------------
    @staticmethod
    def calculate_capability_indices(data: List[float], usl: float, lsl: float, target: float | None = None) -> Dict[str, Any]:
        if not data:
            return {"error": "데이터가 없습니다."}
        if usl <= lsl:
            return {"error": "USL 은 LSL 보다 커야 합니다."}
        arr = np.asarray(data, dtype=float)
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        if std == 0:
            return {"error": "표준편차가 0입니다."}
        cp = (usl - lsl) / (6 * std)
        cpu, cpl = (usl - mean) / (3 * std), (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)
        pp = (usl - lsl) / (6 * arr.std())
        ppk = min((usl - mean) / (3 * arr.std()), (mean - lsl) / (3 * arr.std()))
        defects = np.sum(arr < lsl) + np.sum(arr > usl)
        defect_rate = defects / len(arr) * 100
        cpm = None
        if target is not None:
            cpm = (usl - lsl) / (6 * np.sqrt(np.mean((arr - target) ** 2)))

        def level(val):
            return "Excellent" if val >= 2 else "Very Good" if val >= 1.67 else "Good" if val >= 1.33 else "Acceptable" if val >= 1 else "Poor"

        return {
            "process_statistics": {"mean": round(mean, 4), "std_dev": round(std, 4), "sample_size": len(arr)},
            "spec_limits": {"usl": usl, "lsl": lsl, "target": target},
            "capability_indices": {"cp": round(cp, 3), "cpu": round(cpu, 3), "cpl": round(cpl, 3), "cpk": round(cpk, 3), "pp": round(pp, 3), "ppk": round(ppk, 3), "cpm": None if cpm is None else round(cpm, 3)},
            "performance": {"defect_rate_percent": round(defect_rate, 4), "yield_percent": round(100 - defect_rate, 4)},
            "interpretation": {"capability_level": level(cpk), "sigma_level": round(cpk * 3, 2)},
        }

    # -----------------------------------------------------
    # 3. 패턴 탐지 (WE Rule 기반)
    # -----------------------------------------------------
    @staticmethod
    def detect_patterns(data: List[float], limits: Dict[str, float]) -> Dict[str, Any]:
        if not data or not limits:
            return {"error": "data 또는 limits 가 없습니다."}
        arr = np.asarray(data, dtype=float)
        ucl, lcl, center = limits.get("ucl"), limits.get("lcl"), limits.get("centerline")
        if ucl is None or lcl is None or center is None:
            return {"error": "limits 에 ucl/lcl/centerline 이 필요합니다."}
        sigma = (ucl - center) / 3
        if sigma == 0:
            return {"error": "sigma 가 0입니다."}

        z = (arr - center) / sigma
        patterns: List[Dict[str, Any]] = []

        # Rule 1
        for i, v in enumerate(arr):
            if v > ucl or v < lcl:
                patterns.append({"rule": 1, "index": i, "value": float(v), "desc": "Point beyond control limits"})

        # Rule 2: 9 연속 같은 편향
        for i in range(len(arr) - 8):
            seg = arr[i:i + 9]
            if np.all(seg > center) or np.all(seg < center):
                patterns.append({"rule": 2, "segment": f"{i}-{i + 8}", "desc": "9 points on one side"})

        # Rule 3: 6 연속 증가/감소
        for i in range(len(arr) - 5):
            seg = arr[i:i + 6]
            if np.all(np.diff(seg) > 0) or np.all(np.diff(seg) < 0):
                patterns.append({"rule": 3, "segment": f"{i}-{i + 5}", "desc": "6 points trending"})

        # Rule 5: 2/3 beyond 2 sigma
        for i in range(len(arr) - 2):
            if np.sum(np.abs(z[i:i + 3]) > 2) >= 2:
                patterns.append({"rule": 5, "segment": f"{i}-{i + 2}", "desc": "2 of 3 >2σ"})

        return {
            "total_patterns": len(patterns),
            "patterns": patterns,
            "process_stable": len(patterns) == 0,
        }

    # -----------------------------------------------------
    # 4. Gauge R&R (2 반복 기준 간단법)
    # -----------------------------------------------------
    @staticmethod
    def calculate_gauge_rr(operator_data: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        if not operator_data:
            return {"error": "operator_data 가 비었습니다."}
        all_vals, op_means, op_ranges = [], {}, {}
        for op, trials in operator_data.items():
            trial_means = [np.mean(t) for t in trials if len(t) >= 2]
            trial_ranges = [max(t) - min(t) for t in trials if len(t) >= 2]
            if trial_means:
                op_means[op] = float(np.mean(trial_means))
                op_ranges[op] = float(np.mean(trial_ranges))
            for t in trials:
                all_vals.extend(t)
        if not all_vals:
            return {"error": "유효한 측정값이 없습니다."}
        grand_avg = float(np.mean(all_vals))
        avg_range = float(np.mean(list(op_ranges.values()))) if op_ranges else 0.0
        repeat_std = avg_range / 1.128 if avg_range else 0.0
        repro_std = float(np.std(list(op_means.values()), ddof=1)) if len(op_means) > 1 else 0.0
        rr_std = np.hypot(repeat_std, repro_std)
        total_std = float(np.std(all_vals, ddof=1))
        percent_rr = rr_std / total_std * 100 if total_std else 100
        rating = "Excellent" if percent_rr < 10 else "Acceptable" if percent_rr < 30 else "Unacceptable"
        return {
            "grand_average": round(grand_avg, 4),
            "std_components": {"repeatability": round(repeat_std, 4), "reproducibility": round(repro_std, 4), "rr_std": round(rr_std, 4), "total_std": round(total_std, 4)},
            "percent_rr": round(percent_rr, 2),
            "rating": rating,
        }


# ------------------------------------------------------------
# MCP 래퍼
# ------------------------------------------------------------
@mcp.tool("calculate_control_limits")
def calculate_control_limits(data: str, chart_type: str = "xbar") -> Dict[str, Any]:
    try:
        return ProcessControlAnalyzer.calculate_control_limits(json.loads(data), chart_type)
    except json.JSONDecodeError:
        return {"error": "data JSON 형식 오류"}


@mcp.tool("calculate_capability_indices")
def calculate_capability_indices(data: str, usl: float, lsl: float, target: float | None = None) -> Dict[str, Any]:
    try:
        return ProcessControlAnalyzer.calculate_capability_indices(json.loads(data), usl, lsl, target)
    except json.JSONDecodeError:
        return {"error": "data JSON 형식 오류"}


@mcp.tool("detect_control_patterns")
def detect_control_patterns(data: str, ucl: float, lcl: float, centerline: float) -> Dict[str, Any]:
    try:
        return ProcessControlAnalyzer.detect_patterns(json.loads(data), {"ucl": ucl, "lcl": lcl, "centerline": centerline})
    except json.JSONDecodeError:
        return {"error": "data JSON 형식 오류"}


@mcp.tool("calculate_gauge_rr")
def calculate_gauge_rr(operator_data: str) -> Dict[str, Any]:
    try:
        return ProcessControlAnalyzer.calculate_gauge_rr(json.loads(operator_data))
    except json.JSONDecodeError:
        return {"error": "operator_data JSON 형식 오류"}


# ------------------------------------------------------------
# 실행 스크립트
# ------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Starting Process Control Charts MCP server on port {SERVER_PORT}…")
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=SERVER_PORT, log_level="info")