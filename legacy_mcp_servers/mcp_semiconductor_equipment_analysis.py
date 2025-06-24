# -*- coding: utf-8 -*-
"""
MCP Tool ▶ Semiconductor Equipment Analysis
반도체 설비의 OEE·MTBF/MTTR·가동률·챔버 매칭 등을 분석하기 위한 MCP 툴.
깨진 한글과 문법 오류를 복구하고, 입력 검증·예외 처리를 강화했습니다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import uvicorn
from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------
# 서버 설정
# ------------------------------------------------------------
SERVER_PORT = int(os.getenv("SERVER_PORT", "8010"))

mcp = FastMCP("Semiconductor Equipment Analysis")

# ------------------------------------------------------------
# 핵심 로직 클래스
# ------------------------------------------------------------
class EquipmentAnalyzer:
    """OEE · 신뢰성 · Utilization · Tool Performance 분석용 유틸리티"""

    # -----------------------------------------------------------------
    # 1. OEE
    # -----------------------------------------------------------------
    @staticmethod
    def calculate_oee(
        planned_time: float,
        downtime: float,
        cycle_time: float,
        ideal_cycle_time: float,
        good_units: int,
        total_units: int,
    ) -> Dict[str, Any]:
        """OEE(종합 설비 효율) 계산"""
        if planned_time <= 0 or ideal_cycle_time <= 0 or total_units <= 0:
            return {"error": "planned_time·ideal_cycle_time·total_units 는 0보다 커야 합니다."}
        if downtime < 0 or cycle_time <= 0 or good_units < 0 or good_units > total_units:
            return {"error": "입력 값이 유효 범위를 벗어났습니다."}

        operating_time = planned_time - downtime
        availability = operating_time / planned_time
        performance = ideal_cycle_time / cycle_time
        quality = good_units / total_units
        oee = availability * performance * quality

        def classify(val: float) -> str:
            return "World Class" if val >= 0.85 else "Good" if val >= 0.70 else "Average" if val >= 0.60 else "Poor"

        availability_loss = (1 - availability) * 100
        performance_loss = (1 - performance) * availability * 100
        quality_loss = (1 - quality) * availability * performance * 100

        return {
            "oee_metrics": {
                "oee_percent": round(oee * 100, 2),
                "availability_percent": round(availability * 100, 2),
                "performance_percent": round(performance * 100, 2),
                "quality_percent": round(quality * 100, 2),
            },
            "time_analysis": {
                "planned_time_hours": planned_time,
                "downtime_hours": downtime,
                "operating_time_hours": operating_time,
                "actual_cycle_time_sec": cycle_time,
                "ideal_cycle_time_sec": ideal_cycle_time,
            },
            "production_analysis": {
                "good_units": good_units,
                "total_units": total_units,
                "defective_units": total_units - good_units,
                "defect_rate_percent": round((1 - quality) * 100, 2),
            },
            "loss_analysis": {
                "availability_loss_percent": round(availability_loss, 2),
                "performance_loss_percent": round(performance_loss, 2),
                "quality_loss_percent": round(quality_loss, 2),
                "total_loss_percent": round((1 - oee) * 100, 2),
            },
            "classifications": {
                "oee_class": classify(oee),
                "availability_class": classify(availability),
                "performance_class": classify(performance),
                "quality_class": classify(quality),
            },
            "improvement_priority": max(
                [
                    ("Availability", availability_loss),
                    ("Performance", performance_loss),
                    ("Quality", quality_loss),
                ],
                key=lambda x: x[1],
            )[0],
        }

    # -----------------------------------------------------------------
    # 2. MTBF / MTTR
    # -----------------------------------------------------------------
    @staticmethod
    def calculate_mtbf_mttr(
        failure_times: List[str],
        repair_times: List[str],
    ) -> Dict[str, Any]:
        """MTBF / MTTR 및 Availability 계산"""
        if not failure_times or not repair_times:
            return {"error": "failure_times·repair_times 가 필요합니다."}
        if len(failure_times) != len(repair_times):
            return {"error": "failure_times 와 repair_times 길이가 일치해야 합니다."}

        try:
            fails = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in failure_times]
            repairs = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in repair_times]
        except ValueError as exc:
            return {"error": f"ISO8601 날짜 형식 오류: {exc}"}

        repair_durations = [max((r - f).total_seconds() / 3600, 0) for f, r in zip(fails, repairs)]
        mttr = float(np.mean(repair_durations)) if repair_durations else 0

        if len(fails) > 1:
            intervals = [max((fails[i] - repairs[i - 1]).total_seconds() / 3600, 0) for i in range(1, len(fails))]
            mtbf = float(np.mean(intervals)) if intervals else 0
        else:
            mtbf = 0.0

        availability = mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 0
        failure_rate = 1 / mtbf if mtbf > 0 else float("inf")

        def cls_mtbf(v): return "Excellent" if v >= 500 else "Good" if v >= 200 else "Average" if v >= 100 else "Poor"
        def cls_mttr(v): return "Excellent" if v <= 2 else "Good" if v <= 4 else "Average" if v <= 8 else "Poor"

        return {
            "reliability_metrics": {
                "mtbf_hours": round(mtbf, 2),
                "mttr_hours": round(mttr, 2),
                "availability_percent": round(availability * 100, 2),
                "failure_rate_per_hour": None if failure_rate == float("inf") else round(failure_rate, 6),
            },
            "failure_analysis": {
                "total_failures": len(failure_times),
                "total_repair_time_hours": round(sum(repair_durations), 2),
                "average_repair_time_hours": round(mttr, 2),
                "min_repair_time_hours": round(min(repair_durations), 2) if repair_durations else 0,
                "max_repair_time_hours": round(max(repair_durations), 2) if repair_durations else 0,
            },
            "time_analysis": {
                "analysis_period_hours": round((fails[-1] - fails[0]).total_seconds() / 3600, 2) if len(fails) > 1 else 0,
                "uptime_percent": round(availability * 100, 2),
                "downtime_percent": round((1 - availability) * 100, 2),
            },
            "classifications": {
                "mtbf_class": cls_mtbf(mtbf),
                "mttr_class": cls_mttr(mttr),
                "overall_reliability": cls_mtbf(mtbf) if mtbf > mttr else cls_mttr(mttr),
            },
        }

    # -----------------------------------------------------------------
    # 3. Equipment Utilization
    # -----------------------------------------------------------------
    @staticmethod
    def analyze_equipment_utilization(production_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """가동률·유휴·셋업·유지보수 시간을 분석"""
        if not production_data:
            return {"error": "production_data 가 필요합니다."}

        totals = {"production": 0.0, "setup": 0.0, "idle": 0.0, "maintenance": 0.0}
        total_time = 0.0
        units_list: List[int] = []

        for rec in production_data:
            try:
                dur = float(rec.get("duration_hours", 0))
                status = str(rec.get("status", "")).lower()
                units = int(rec.get("units_produced", 0))
            except (TypeError, ValueError):
                continue
            total_time += dur
            units_list.append(units)
            if status in {"production", "running"}:
                totals["production"] += dur
            elif status in {"setup", "changeover"}:
                totals["setup"] += dur
            elif status in {"idle", "waiting"}:
                totals["idle"] += dur
            elif status in {"maintenance", "repair"}:
                totals["maintenance"] += dur

        if total_time == 0:
            return {"error": "총 시간이 0입니다."}

        util_pct = {k: round(v / total_time * 100, 2) for k, v in totals.items()}
        total_units = sum(units_list)
        avg_units_per_hr = total_units / total_time
        consistency = round(np.std(units_list) / np.mean(units_list), 2) if units_list and np.mean(units_list) > 0 else 0

        classification = (
            "Excellent" if util_pct["production"] >= 80 else "Good" if util_pct["production"] >= 70 else "Average" if util_pct["production"] >= 60 else "Poor"
        )

        return {
            "time_breakdown": {
                "total_time_hours": round(total_time, 2),
                "production_time_hours": round(totals["production"], 2),
                "setup_time_hours": round(totals["setup"], 2),
                "idle_time_hours": round(totals["idle"], 2),
                "maintenance_time_hours": round(totals["maintenance"], 2),
            },
            "utilization_percentages": util_pct,
            "productivity_metrics": {
                "total_units": total_units,
                "average_units_per_hour": round(avg_units_per_hr, 2),
                "peak_production_units": max(units_list) if units_list else 0,
                "production_consistency_cv": consistency,
            },
            "efficiency_analysis": {
                "effectiveness_percent": round((totals["production"] + totals["setup"]) / total_time * 100, 2),
                "waste_time_percent": round((totals["idle"] + totals["maintenance"]) / total_time * 100, 2),
            },
            "classification": classification,
            "improvement_flags": {
                "reduce_idle_time": totals["idle"] > total_time * 0.1,
                "optimize_setup": totals["setup"] > total_time * 0.15,
                "improve_maintenance": totals["maintenance"] > total_time * 0.05,
            },
        }

    # -----------------------------------------------------------------
    # 4. Tool Performance
    # -----------------------------------------------------------------
    @staticmethod
    def analyze_tool_performance(
        tool_metrics: Dict[str, List[float]],
        tool_specs: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """측정 지표별 Cp/Cpk·Trend·Target 편차 분석"""
        if not tool_metrics or not tool_specs:
            return {"error": "tool_metrics·tool_specs 가 필요합니다."}

        analyses: Dict[str, Any] = {}
        cpks, yields = [], []

        for name, vals in tool_metrics.items():
            if not vals:
                continue
            arr = np.asarray(vals)
            mean, std = float(arr.mean()), float(arr.std())
            specs = tool_specs.get(name, {})
            lsl, usl, target = specs.get("lsl"), specs.get("usl"), specs.get("target")

            perf = {
                "mean": round(mean, 4),
                "std_dev": round(std, 4),
                "min": round(arr.min(), 4),
                "max": round(arr.max(), 4),
                "cv_percent": round(std / mean * 100, 2) if mean else None,
            }

            spec_res: Dict[str, Any] = {}
            if lsl is not None and usl is not None and std > 0:
                within = np.sum((arr >= lsl) & (arr <= usl))
                spec_yield = within / len(arr) * 100
                cp = (usl - lsl) / (6 * std)
                cpu = (usl - mean) / (3 * std)
                cpl = (mean - lsl) / (3 * std)
                cpk = min(cpu, cpl)
                spec_res = {
                    "spec_yield_percent": round(spec_yield, 2),
                    "cp": round(cp, 3),
                    "cpk": round(cpk, 3),
                    "within_spec_count": int(within),
                    "out_of_spec_count": int(len(arr) - within),
                    "usl": usl,
                    "lsl": lsl,
                }
                cpks.append(cpk)
                yields.append(spec_yield)

            target_res = {}
            if target is not None:
                bias = mean - target
                target_res = {
                    "target": target,
                    "bias": round(bias, 4),
                    "bias_percent": round(bias / target * 100, 2) if target else None,
                }

            if len(arr) > 1:
                x = np.arange(len(arr))
                slope, _ = np.polyfit(x, arr, 1)
                trend_dir = "Increasing" if slope > 0.001 else "Decreasing" if slope < -0.001 else "Stable"
                trend_res = {"trend_slope": round(slope, 6), "trend_direction": trend_dir}
            else:
                trend_res = {"trend_direction": "Insufficient data"}

            analyses[name] = {
                "performance_metrics": perf,
                "specification_analysis": spec_res,
                "target_analysis": target_res,
                "trend_analysis": trend_res,
                "data_points": len(arr),
            }

        summary = {
            "total_metrics": len(analyses),
            "average_cpk": round(np.mean(cpks), 3) if cpks else None,
            "average_yield_percent": round(np.mean(yields), 2) if yields else None,
            "tool_performance_grade": "A" if cpks and np.mean(cpks) >= 1.33 else "B" if cpks and np.mean(cpks) >= 1.0 else "C",
        }
        return {"tool_analysis": analyses, "overall_summary": summary}

    # -----------------------------------------------------------------
    # 5. Chamber Matching
    # -----------------------------------------------------------------
    @staticmethod
    def calculate_chamber_matching(chamber_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """챔버 간 성능 균일도 분석"""
        if not chamber_data:
            return {"error": "chamber_data 가 필요합니다."}

        chamber_stats: Dict[str, Any] = {}
        all_vals: List[float] = []
        for cid, vals in chamber_data.items():
            if not vals:
                continue
            arr = np.asarray(vals)
            chamber_stats[cid] = {
                "mean": round(float(arr.mean()), 4),
                "std_dev": round(float(arr.std()), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "count": int(len(arr)),
            }
            all_vals.extend(arr.tolist())

        if not all_vals:
            return {"error": "유효한 데이터가 없습니다."}

        all_arr = np.asarray(all_vals)
        overall_mean, overall_std = float(all_arr.mean()), float(all_arr.std())
        means = np.asarray([s["mean"] for s in chamber_stats.values()])
        between_std = float(means.std())
        within_std = float(np.mean([s["std_dev"] for s in chamber_stats.values()]))
        matching_ratio = within_std / between_std if between_std > 0 else float("inf")

        grade = (
            "Excellent" if between_std / overall_mean < 0.01 else "Good" if between_std / overall_mean < 0.03 else "Fair" if between_std / overall_mean < 0.05 else "Poor"
        )

        outliers = []
        for cid, stats in chamber_stats.items():
            if overall_std == 0:
                continue
            z = abs(stats["mean"] - overall_mean) / overall_std
            if z > 2:
                outliers.append({"chamber_id": cid, "z_score": round(float(z), 2)})

        return {
            "chamber_statistics": chamber_stats,
            "overall_statistics": {
                "overall_mean": round(overall_mean, 4),
                "overall_std": round(overall_std, 4),
            },
            "matching_analysis": {
                "between_chamber_std": round(between_std, 4),
                "within_chamber_std": round(within_std, 4),
                "matching_ratio": None if matching_ratio == float("inf") else round(matching_ratio, 3),
                "matching_grade": grade,
            },
            "outlier_chambers": outliers,
        }


# ------------------------------------------------------------
# MCP 래퍼 함수들
# ------------------------------------------------------------
@mcp.tool("calculate_oee")
def calculate_oee(
    planned_time: float,
    downtime: float,
    cycle_time: float,
    ideal_cycle_time: float,
    good_units: int,
    total_units: int,
) -> Dict[str, Any]:
    """OEE 계산"""
    return EquipmentAnalyzer.calculate_oee(planned_time, downtime, cycle_time, ideal_cycle_time, good_units, total_units)


@mcp.tool("calculate_mtbf_mttr")
def calculate_mtbf_mttr(failure_times: str, repair_times: str) -> Dict[str, Any]:
    """MTBF / MTTR 계산"""
    try:
        return EquipmentAnalyzer.calculate_mtbf_mttr(json.loads(failure_times), json.loads(repair_times))
    except json.JSONDecodeError:
        return {"error": "JSON 형식 오류"}


@mcp.tool("analyze_equipment_utilization")
def analyze_equipment_utilization(production_data: str) -> Dict[str, Any]:
    """Equipment Utilization 분석"""
    try:
        return EquipmentAnalyzer.analyze_equipment_utilization(json.loads(production_data))
    except json.JSONDecodeError:
        return {"error": "JSON 형식 오류"}


@mcp.tool("analyze_tool_performance")
def analyze_tool_performance(tool_metrics: str, tool_specs: str) -> Dict[str, Any]:
    """Tool Performance 분석"""
    try:
        return EquipmentAnalyzer.analyze_tool_performance(json.loads(tool_metrics), json.loads(tool_specs))
    except json.JSONDecodeError:
        return {"error": "JSON 형식 오류"}


@mcp.tool("calculate_chamber_matching")
def calculate_chamber_matching(chamber_data: str) -> Dict[str, Any]:
    """Chamber Matching 분석"""
    try:
        return EquipmentAnalyzer.calculate_chamber_matching(json.loads(chamber_data))
    except json.JSONDecodeError:
        return {"error": "JSON 형식 오류"}


# ------------------------------------------------------------
# 실행 스크립트
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting Semiconductor Equipment Analysis MCP server on port {SERVER_PORT}…")
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=SERVER_PORT, log_level="info")
s