#!/usr/bin/env python3
"""
MCP Tool: Process Control Charts
공정 관리도 분석 도구 - SPC, Cp/Cpk, 관리한계 분석
"""

from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import math
import json
import os
import uvicorn

# Get port from environment variable
SERVER_PORT = int(os.getenv('SERVER_PORT', '8009'))

# FastMCP 서버 생성
mcp = FastMCP("Process Control Charts")

class ProcessControlAnalyzer:
    """공정 관리도 분석기"""
    
    @staticmethod
    def calculate_control_limits(data: List[float], chart_type: str = "xbar") -> Dict[str, Any]:
        """관리한계 계산"""
        if not data or len(data) < 2:
            return {"error": "Insufficient data points"}
        
        data_array = np.array(data)
        n = len(data_array)
        
        if chart_type.lower() == "xbar":
            # X-bar 관리도
            centerline = np.mean(data_array)
            
            # 이동범위 계산
            moving_ranges = [abs(data_array[i] - data_array[i-1]) for i in range(1, n)]
            avg_moving_range = np.mean(moving_ranges) if moving_ranges else 0
            
            # 관리한계 (d2=1.128 for n=2)
            d2 = 1.128
            ucl = centerline + (3 * avg_moving_range / d2)
            lcl = centerline - (3 * avg_moving_range / d2)
            
            # 경고한계 (2σ)
            warning_ucl = centerline + (2 * avg_moving_range / d2)
            warning_lcl = centerline - (2 * avg_moving_range / d2)
            
        elif chart_type.lower() == "r":
            # R 관리도 (Range Chart)
            if n < 2:
                return {"error": "Need at least 2 points for R chart"}
            
            ranges = [abs(data_array[i] - data_array[i-1]) for i in range(1, n)]
            centerline = np.mean(ranges)
            
            # R 관리도 상수 (D3, D4 for n=2)
            D3, D4 = 0, 3.267
            ucl = D4 * centerline
            lcl = D3 * centerline
            
            warning_ucl = centerline + (2/3) * (ucl - centerline)
            warning_lcl = max(0, centerline - (2/3) * (centerline - lcl))
            
        else:
            return {"error": f"Unsupported chart type: {chart_type}"}
        
        # 관리상태 판정
        out_of_control = []
        for i, value in enumerate(data_array):
            if value > ucl or value < lcl:
                out_of_control.append({
                    "index": i,
                    "value": value,
                    "type": "beyond_limits"
                })
        
        return {
            "chart_type": chart_type,
            "centerline": round(centerline, 4),
            "ucl": round(ucl, 4),
            "lcl": round(lcl, 4),
            "warning_ucl": round(warning_ucl, 4),
            "warning_lcl": round(warning_lcl, 4),
            "out_of_control_points": out_of_control,
            "process_in_control": len(out_of_control) == 0,
            "data_points": len(data),
            "process_variation": round(avg_moving_range if chart_type.lower() == "xbar" else np.std(ranges), 4)
        }
    
    @staticmethod
    def calculate_capability_indices(data: List[float], usl: float, lsl: float, target: float = None) -> Dict[str, Any]:
        """공정능력지수 (Cp, Cpk) 계산"""
        if not data:
            return {"error": "No data provided"}
        
        if usl <= lsl:
            return {"error": "USL must be greater than LSL"}
        
        data_array = np.array(data)
        process_mean = np.mean(data_array)
        process_std = np.std(data_array, ddof=1)  # 표본 표준편차
        
        if process_std == 0:
            return {"error": "Process standard deviation is zero"}
        
        # 공정능력지수 계산
        cp = (usl - lsl) / (6 * process_std)
        cpu = (usl - process_mean) / (3 * process_std)
        cpl = (process_mean - lsl) / (3 * process_std)
        cpk = min(cpu, cpl)
        
        # 공정성능지수 (Pp, Ppk) - 전체 모집단 기준
        pp = (usl - lsl) / (6 * np.std(data_array))  # 모집단 표준편차
        ppk = min((usl - process_mean) / (3 * np.std(data_array)), 
                  (process_mean - lsl) / (3 * np.std(data_array)))
        
        # 불량률 계산
        below_lsl = np.sum(data_array < lsl)
        above_usl = np.sum(data_array > usl)
        defect_rate = (below_lsl + above_usl) / len(data_array) * 100
        
        # 타겟 대비 성능 (Cpm)
        cpm = None
        if target is not None:
            target_variance = np.mean((data_array - target) ** 2)
            cpm = (usl - lsl) / (6 * np.sqrt(target_variance))
        
        # 해석
        def interpret_capability(cpk_value):
            if cpk_value >= 2.0:
                return "Excellent (6σ)"
            elif cpk_value >= 1.67:
                return "Very Good (5σ)"
            elif cpk_value >= 1.33:
                return "Good (4σ)"
            elif cpk_value >= 1.0:
                return "Acceptable (3σ)"
            else:
                return "Poor (<3σ)"
        
        return {
            "process_statistics": {
                "mean": round(process_mean, 4),
                "std_dev": round(process_std, 4),
                "sample_size": len(data_array)
            },
            "specification_limits": {
                "usl": usl,
                "lsl": lsl,
                "target": target,
                "tolerance": usl - lsl
            },
            "capability_indices": {
                "cp": round(cp, 3),
                "cpk": round(cpk, 3),
                "cpu": round(cpu, 3),
                "cpl": round(cpl, 3),
                "pp": round(pp, 3),
                "ppk": round(ppk, 3),
                "cpm": round(cpm, 3) if cpm is not None else None
            },
            "performance_metrics": {
                "defect_rate_percent": round(defect_rate, 4),
                "yield_percent": round(100 - defect_rate, 4),
                "below_lsl_count": int(below_lsl),
                "above_usl_count": int(above_usl)
            },
            "interpretation": {
                "capability_level": interpret_capability(cpk),
                "process_centered": abs(cpu - cpl) < 0.1,
                "sigma_level": round(cpk * 3, 1)
            }
        }
    
    @staticmethod
    def detect_patterns(data: List[float], control_limits: Dict[str, float]) -> Dict[str, Any]:
        """관리도 패턴 탐지 (웨스턴 일렉트릭 룰)"""
        if not data or not control_limits:
            return {"error": "Invalid data or control limits"}
        
        data_array = np.array(data)
        ucl = control_limits.get("ucl", 0)
        lcl = control_limits.get("lcl", 0)
        centerline = control_limits.get("centerline", 0)
        
        # 표준화된 값 계산
        sigma = (ucl - centerline) / 3
        if sigma == 0:
            return {"error": "Invalid control limits - sigma is zero"}
        
        z_scores = (data_array - centerline) / sigma
        
        patterns = []
        
        # Rule 1: 관리한계를 벗어난 점
        for i, value in enumerate(data_array):
            if value > ucl or value < lcl:
                patterns.append({
                    "rule": "Rule 1",
                    "description": "Point beyond control limits",
                    "index": i,
                    "value": value,
                    "severity": "High"
                })
        
        # Rule 2: 연속 9점이 중심선 한쪽에 위치
        for i in range(len(data_array) - 8):
            segment = data_array[i:i+9]
            if all(x > centerline for x in segment) or all(x < centerline for x in segment):
                patterns.append({
                    "rule": "Rule 2",
                    "description": "9 consecutive points on one side of centerline",
                    "index": i,
                    "segment": f"{i} to {i+8}",
                    "severity": "Medium"
                })
        
        # Rule 3: 연속 6점이 증가 또는 감소
        for i in range(len(data_array) - 5):
            segment = data_array[i:i+6]
            if all(segment[j] < segment[j+1] for j in range(5)) or \
               all(segment[j] > segment[j+1] for j in range(5)):
                patterns.append({
                    "rule": "Rule 3",
                    "description": "6 consecutive increasing or decreasing points",
                    "index": i,
                    "segment": f"{i} to {i+5}",
                    "severity": "Medium"
                })
        
        # Rule 4: 연속 14점이 교대로 증감
        for i in range(len(data_array) - 13):
            segment = data_array[i:i+14]
            alternating_up = all((segment[j+1] - segment[j]) * (segment[j] - segment[j-1] if j > 0 else 1) < 0 
                               for j in range(1, 13))
            if alternating_up:
                patterns.append({
                    "rule": "Rule 4",
                    "description": "14 consecutive alternating points",
                    "index": i,
                    "segment": f"{i} to {i+13}",
                    "severity": "Low"
                })
        
        # Rule 5: 3점 중 2점이 2σ 경계 밖에 위치
        for i in range(len(data_array) - 2):
            segment = z_scores[i:i+3]
            beyond_2sigma = sum(1 for z in segment if abs(z) > 2)
            if beyond_2sigma >= 2:
                patterns.append({
                    "rule": "Rule 5",
                    "description": "2 out of 3 points beyond 2σ",
                    "index": i,
                    "segment": f"{i} to {i+2}",
                    "severity": "Medium"
                })
        
        # 패턴 심각도별 분류
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for pattern in patterns:
            severity_counts[pattern["severity"]] += 1
        
        return {
            "total_patterns": len(patterns),
            "patterns_by_severity": severity_counts,
            "detailed_patterns": patterns,
            "process_stable": len(patterns) == 0,
            "recommendation": "Investigate assignable causes" if patterns else "Process appears stable"
        }

@mcp.tool()
def calculate_control_limits(data: str, chart_type: str = "xbar") -> Dict[str, Any]:
    """
    관리한계 계산
    
    Args:
        data: 측정 데이터 (JSON 배열 문자열)
        chart_type: 관리도 타입 ("xbar" 또는 "r")
    
    Returns:
        관리한계 및 관리상태 분석 결과
    """
    try:
        data_list = json.loads(data)
        return ProcessControlAnalyzer.calculate_control_limits(data_list, chart_type)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for data"}

@mcp.tool()
def calculate_capability_indices(data: str, usl: float, lsl: float, target: float = None) -> Dict[str, Any]:
    """
    공정능력지수 계산
    
    Args:
        data: 측정 데이터 (JSON 배열 문자열)
        usl: 상한 규격 (Upper Specification Limit)
        lsl: 하한 규격 (Lower Specification Limit)
        target: 목표값 (선택사항)
    
    Returns:
        공정능력지수 (Cp, Cpk, Pp, Ppk) 및 성능 메트릭
    """
    try:
        data_list = json.loads(data)
        return ProcessControlAnalyzer.calculate_capability_indices(data_list, usl, lsl, target)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for data"}

@mcp.tool()
def detect_control_patterns(data: str, ucl: float, lcl: float, centerline: float) -> Dict[str, Any]:
    """
    관리도 패턴 탐지 (웨스턴 일렉트릭 룰)
    
    Args:
        data: 측정 데이터 (JSON 배열 문자열)
        ucl: 상위 관리한계
        lcl: 하위 관리한계
        centerline: 중심선
    
    Returns:
        탐지된 패턴 및 이상상태 분석
    """
    try:
        data_list = json.loads(data)
        control_limits = {"ucl": ucl, "lcl": lcl, "centerline": centerline}
        return ProcessControlAnalyzer.detect_patterns(data_list, control_limits)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for data"}

@mcp.tool()
def analyze_spc_run(data: str, subgroup_size: int = 5) -> Dict[str, Any]:
    """
    SPC 런 분석 (연속 생산 데이터)
    
    Args:
        data: 측정 데이터 (JSON 배열 문자열)
        subgroup_size: 부분군 크기
    
    Returns:
        종합적인 SPC 분석 결과
    """
    try:
        data_list = json.loads(data)
        
        if len(data_list) < subgroup_size:
            return {"error": f"Need at least {subgroup_size} data points"}
        
        # 부분군별 분석
        subgroups = [data_list[i:i+subgroup_size] for i in range(0, len(data_list), subgroup_size)]
        if len(subgroups[-1]) < subgroup_size:
            subgroups = subgroups[:-1]  # 마지막 불완전한 부분군 제거
        
        subgroup_means = [np.mean(sg) for sg in subgroups]
        subgroup_ranges = [max(sg) - min(sg) for sg in subgroups]
        
        # X-bar 관리도
        xbar_limits = ProcessControlAnalyzer.calculate_control_limits(subgroup_means, "xbar")
        
        # R 관리도
        r_limits = ProcessControlAnalyzer.calculate_control_limits(subgroup_ranges, "r")
        
        # 전체 데이터에 대한 능력분석 (가상의 규격한계 사용)
        overall_mean = np.mean(data_list)
        overall_std = np.std(data_list)
        virtual_usl = overall_mean + 3 * overall_std
        virtual_lsl = overall_mean - 3 * overall_std
        
        capability = ProcessControlAnalyzer.calculate_capability_indices(
            data_list, virtual_usl, virtual_lsl, overall_mean)
        
        return {
            "subgroup_analysis": {
                "total_subgroups": len(subgroups),
                "subgroup_size": subgroup_size,
                "subgroup_means": [round(m, 3) for m in subgroup_means],
                "subgroup_ranges": [round(r, 3) for r in subgroup_ranges]
            },
            "xbar_chart": xbar_limits,
            "r_chart": r_limits,
            "overall_statistics": {
                "grand_mean": round(overall_mean, 4),
                "overall_std": round(overall_std, 4),
                "total_data_points": len(data_list)
            },
            "process_capability": capability["capability_indices"] if "error" not in capability else None,
            "process_status": "In Control" if (xbar_limits.get("process_in_control", False) and 
                                             r_limits.get("process_in_control", False)) else "Out of Control"
        }
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for data"}

@mcp.tool()
def calculate_gauge_rr(operator_data: str) -> Dict[str, Any]:
    """
    게이지 R&R 분석
    
    Args:
        operator_data: 운영자별 측정 데이터 (JSON 객체 문자열)
                      형식: {"operator1": [[trial1_measurements], [trial2_measurements]], ...}
    
    Returns:
        게이지 R&R 분석 결과
    """
    try:
        data_dict = json.loads(operator_data)
        
        if not data_dict:
            return {"error": "No operator data provided"}
        
        all_measurements = []
        operator_averages = {}
        operator_ranges = {}
        
        for operator, trials in data_dict.items():
            trial_averages = []
            trial_ranges = []
            
            for trial in trials:
                if len(trial) < 2:
                    continue
                trial_avg = np.mean(trial)
                trial_range = max(trial) - min(trial)
                trial_averages.append(trial_avg)
                trial_ranges.append(trial_range)
                all_measurements.extend(trial)
            
            operator_averages[operator] = np.mean(trial_averages) if trial_averages else 0
            operator_ranges[operator] = np.mean(trial_ranges) if trial_ranges else 0
        
        # 전체 평균 및 범위
        grand_average = np.mean(all_measurements) if all_measurements else 0
        average_range = np.mean(list(operator_ranges.values())) if operator_ranges else 0
        
        # 재현성 (Repeatability) - 장비 변동
        repeatability_std = average_range / 1.128  # d2 constant for n=2
        
        # 재생성 (Reproducibility) - 운영자 변동
        operator_variation = np.std(list(operator_averages.values())) if len(operator_averages) > 1 else 0
        reproducibility_std = operator_variation
        
        # R&R 계산
        rr_variance = repeatability_std**2 + reproducibility_std**2
        rr_std = np.sqrt(rr_variance)
        
        # 전체 변동 대비 비율
        total_std = np.std(all_measurements) if all_measurements else 1
        part_to_part_variance = max(0, total_std**2 - rr_variance)
        part_to_part_std = np.sqrt(part_to_part_variance)
        
        # %R&R 계산
        percent_rr = (rr_std / total_std) * 100 if total_std > 0 else 100
        percent_repeatability = (repeatability_std / total_std) * 100 if total_std > 0 else 100
        percent_reproducibility = (reproducibility_std / total_std) * 100 if total_std > 0 else 100
        
        # 판정
        if percent_rr < 10:
            rr_rating = "Excellent"
        elif percent_rr < 30:
            rr_rating = "Acceptable"
        else:
            rr_rating = "Unacceptable"
        
        return {
            "operators": list(data_dict.keys()),
            "measurements_count": len(all_measurements),
            "grand_average": round(grand_average, 4),
            "variance_components": {
                "repeatability_std": round(repeatability_std, 4),
                "reproducibility_std": round(reproducibility_std, 4),
                "rr_std": round(rr_std, 4),
                "part_to_part_std": round(part_to_part_std, 4),
                "total_std": round(total_std, 4)
            },
            "percent_study_variation": {
                "percent_rr": round(percent_rr, 2),
                "percent_repeatability": round(percent_repeatability, 2),
                "percent_reproducibility": round(percent_reproducibility, 2),
                "percent_part_to_part": round((part_to_part_std/total_std)*100, 2)
            },
            "rating": rr_rating,
            "acceptable": percent_rr < 30,
            "operator_averages": {k: round(v, 4) for k, v in operator_averages.items()},
            "recommendation": f"게이지 R&R {percent_rr:.1f}% - {rr_rating}"
        }
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for operator data"}

if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Process Control Charts MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)