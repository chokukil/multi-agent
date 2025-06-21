#!/usr/bin/env python3
"""
MCP Tool: Semiconductor Equipment Analysis
반도체 장비 분석 도구 - OEE, MTBF/MTTR, 가동률, 장비 성능 분석
"""

import os
import uvicorn
from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import json

# 환경변수에서 포트 가져오기 (기본값: 8010)
SERVER_PORT = int(os.getenv('SERVER_PORT', '8010'))

# FastMCP 서버 생성
mcp = FastMCP("Semiconductor Equipment Analysis")

class EquipmentAnalyzer:
    """반도체 장비 분석기"""
    
    @staticmethod
    def calculate_oee(planned_time: float, downtime: float, cycle_time: float, 
                     ideal_cycle_time: float, good_units: int, total_units: int) -> Dict[str, Any]:
        """OEE (Overall Equipment Effectiveness) 계산"""
        if planned_time <= 0 or ideal_cycle_time <= 0 or total_units <= 0:
            return {"error": "Invalid input parameters"}
        
        # 가동률 (Availability)
        operating_time = planned_time - downtime
        availability = operating_time / planned_time
        
        # 성능률 (Performance)
        if cycle_time <= 0:
            performance = 0
        else:
            performance = (ideal_cycle_time / cycle_time)
        
        # 품질률 (Quality)
        quality = good_units / total_units if total_units > 0 else 0
        
        # OEE 계산
        oee = availability * performance * quality
        
        # 분류 및 권고사항
        def classify_metric(value, metric_name):
            if value >= 0.85:
                return "World Class"
            elif value >= 0.70:
                return "Good"
            elif value >= 0.60:
                return "Average"
            else:
                return "Poor"
        
        # 손실 분석
        availability_loss = (1 - availability) * 100
        performance_loss = (1 - performance) * availability * 100
        quality_loss = (1 - quality) * availability * performance * 100
        
        return {
            "oee_metrics": {
                "oee_percent": round(oee * 100, 2),
                "availability_percent": round(availability * 100, 2),
                "performance_percent": round(performance * 100, 2),
                "quality_percent": round(quality * 100, 2)
            },
            "time_analysis": {
                "planned_time_hours": planned_time,
                "downtime_hours": downtime,
                "operating_time_hours": operating_time,
                "actual_cycle_time": cycle_time,
                "ideal_cycle_time": ideal_cycle_time
            },
            "production_analysis": {
                "good_units": good_units,
                "total_units": total_units,
                "defective_units": total_units - good_units,
                "defect_rate_percent": round((1 - quality) * 100, 2)
            },
            "loss_analysis": {
                "availability_loss_percent": round(availability_loss, 2),
                "performance_loss_percent": round(performance_loss, 2),
                "quality_loss_percent": round(quality_loss, 2),
                "total_loss_percent": round((1 - oee) * 100, 2)
            },
            "classifications": {
                "oee_class": classify_metric(oee, "OEE"),
                "availability_class": classify_metric(availability, "Availability"),
                "performance_class": classify_metric(performance, "Performance"),
                "quality_class": classify_metric(quality, "Quality")
            },
            "improvement_priority": max(
                [("Availability", availability_loss), ("Performance", performance_loss), ("Quality", quality_loss)],
                key=lambda x: x[1]
            )[0]
        }
    
    @staticmethod
    def calculate_mtbf_mttr(failure_times: List[str], repair_times: List[str]) -> Dict[str, Any]:
        """MTBF (Mean Time Between Failures) 및 MTTR (Mean Time To Repair) 계산"""
        if not failure_times or not repair_times:
            return {"error": "Failure times and repair times required"}
        
        if len(failure_times) != len(repair_times):
            return {"error": "Number of failure times must match repair times"}
        
        try:
            # 시간 파싱
            failure_datetimes = [datetime.fromisoformat(ft.replace('Z', '+00:00')) for ft in failure_times]
            repair_datetimes = [datetime.fromisoformat(rt.replace('Z', '+00:00')) for rt in repair_times]
            
            # MTTR 계산 (수리 시간)
            repair_durations = []
            for i in range(len(failure_datetimes)):
                if repair_datetimes[i] > failure_datetimes[i]:
                    duration = (repair_datetimes[i] - failure_datetimes[i]).total_seconds() / 3600  # 시간 단위
                    repair_durations.append(duration)
            
            mttr = np.mean(repair_durations) if repair_durations else 0
            
            # MTBF 계산 (고장 간격)
            if len(failure_datetimes) > 1:
                failure_intervals = []
                for i in range(1, len(failure_datetimes)):
                    # 이전 수리 완료 시점부터 다음 고장까지의 시간
                    interval = (failure_datetimes[i] - repair_datetimes[i-1]).total_seconds() / 3600
                    if interval > 0:
                        failure_intervals.append(interval)
                
                mtbf = np.mean(failure_intervals) if failure_intervals else 0
            else:
                mtbf = 0
            
            # 가용도 계산
            availability = mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 0
            
            # 신뢰성 지표
            failure_rate = 1 / mtbf if mtbf > 0 else float('inf')  # 시간당 고장률
            
            # 분류
            def classify_mtbf(mtbf_hours):
                if mtbf_hours >= 500:
                    return "Excellent"
                elif mtbf_hours >= 200:
                    return "Good"
                elif mtbf_hours >= 100:
                    return "Average"
                else:
                    return "Poor"
            
            def classify_mttr(mttr_hours):
                if mttr_hours <= 2:
                    return "Excellent"
                elif mttr_hours <= 4:
                    return "Good"
                elif mttr_hours <= 8:
                    return "Average"
                else:
                    return "Poor"
            
            return {
                "reliability_metrics": {
                    "mtbf_hours": round(mtbf, 2),
                    "mttr_hours": round(mttr, 2),
                    "availability_percent": round(availability * 100, 2),
                    "failure_rate_per_hour": round(failure_rate, 6)
                },
                "failure_analysis": {
                    "total_failures": len(failure_times),
                    "total_repair_time_hours": round(sum(repair_durations), 2),
                    "average_repair_time_hours": round(mttr, 2),
                    "min_repair_time_hours": round(min(repair_durations), 2) if repair_durations else 0,
                    "max_repair_time_hours": round(max(repair_durations), 2) if repair_durations else 0
                },
                "time_analysis": {
                    "analysis_period_hours": round((failure_datetimes[-1] - failure_datetimes[0]).total_seconds() / 3600, 2) if len(failure_datetimes) > 1 else 0,
                    "uptime_percent": round(availability * 100, 2),
                    "downtime_percent": round((1 - availability) * 100, 2)
                },
                "classifications": {
                    "mtbf_class": classify_mtbf(mtbf),
                    "mttr_class": classify_mttr(mttr),
                    "overall_reliability": classify_mtbf(mtbf) if mtbf > mttr else classify_mttr(mttr)
                }
            }
            
        except ValueError as e:
            return {"error": f"Invalid datetime format: {str(e)}"}
    
    @staticmethod
    def analyze_equipment_utilization(production_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """장비 가동률 및 활용도 분석"""
        if not production_data:
            return {"error": "No production data provided"}
        
        total_time = 0
        setup_time = 0
        production_time = 0
        idle_time = 0
        maintenance_time = 0
        
        production_counts = []
        
        for record in production_data:
            try:
                duration = record.get('duration_hours', 0)
                status = record.get('status', '').lower()
                units_produced = record.get('units_produced', 0)
                
                total_time += duration
                production_counts.append(units_produced)
                
                if status in ['production', 'running']:
                    production_time += duration
                elif status in ['setup', 'changeover']:
                    setup_time += duration
                elif status in ['idle', 'waiting']:
                    idle_time += duration
                elif status in ['maintenance', 'repair']:
                    maintenance_time += duration
                    
            except (KeyError, TypeError):
                continue
        
        if total_time == 0:
            return {"error": "Total time is zero"}
        
        # 활용도 계산
        utilization_metrics = {
            "production_utilization_percent": round((production_time / total_time) * 100, 2),
            "setup_utilization_percent": round((setup_time / total_time) * 100, 2),
            "idle_utilization_percent": round((idle_time / total_time) * 100, 2),
            "maintenance_utilization_percent": round((maintenance_time / total_time) * 100, 2)
        }
        
        # 생산성 분석
        total_units = sum(production_counts)
        avg_units_per_hour = total_units / total_time if total_time > 0 else 0
        peak_production = max(production_counts) if production_counts else 0
        
        # 효율성 분석
        effective_time = production_time + setup_time  # 부가가치 시간
        effectiveness = effective_time / total_time if total_time > 0 else 0
        
        # 벤치마크 비교
        def classify_utilization(production_util):
            if production_util >= 80:
                return "Excellent"
            elif production_util >= 70:
                return "Good"
            elif production_util >= 60:
                return "Average"
            else:
                return "Poor"
        
        return {
            "time_breakdown": {
                "total_time_hours": round(total_time, 2),
                "production_time_hours": round(production_time, 2),
                "setup_time_hours": round(setup_time, 2),
                "idle_time_hours": round(idle_time, 2),
                "maintenance_time_hours": round(maintenance_time, 2)
            },
            "utilization_metrics": utilization_metrics,
            "productivity_metrics": {
                "total_units_produced": total_units,
                "average_units_per_hour": round(avg_units_per_hour, 2),
                "peak_production_rate": peak_production,
                "production_consistency": round(np.std(production_counts) / np.mean(production_counts), 2) if production_counts and np.mean(production_counts) > 0 else 0
            },
            "efficiency_analysis": {
                "effectiveness_percent": round(effectiveness * 100, 2),
                "waste_time_percent": round(((idle_time + maintenance_time) / total_time) * 100, 2),
                "value_added_time_percent": round((production_time / total_time) * 100, 2)
            },
            "classification": classify_utilization(utilization_metrics["production_utilization_percent"]),
            "improvement_opportunities": {
                "reduce_idle_time": idle_time > total_time * 0.1,
                "optimize_setup": setup_time > total_time * 0.15,
                "improve_maintenance": maintenance_time > total_time * 0.05
            }
        }
    
    @staticmethod
    def analyze_tool_performance(tool_metrics: Dict[str, List[float]], tool_specs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """툴 성능 분석 (예: 에칭율, 증착율, 온도 균일성 등)"""
        if not tool_metrics or not tool_specs:
            return {"error": "Tool metrics and specifications required"}
        
        analysis_results = {}
        
        for metric_name, values in tool_metrics.items():
            if not values:
                continue
                
            specs = tool_specs.get(metric_name, {})
            target = specs.get('target')
            usl = specs.get('usl')  # Upper Specification Limit
            lsl = specs.get('lsl')  # Lower Specification Limit
            
            # 기본 통계
            mean_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)
            
            # 성능 지표
            performance_metrics = {
                "mean": round(mean_value, 4),
                "std_dev": round(std_value, 4),
                "min": round(min_value, 4),
                "max": round(max_value, 4),
                "range": round(max_value - min_value, 4),
                "cv_percent": round((std_value / mean_value) * 100, 2) if mean_value != 0 else 0
            }
            
            # 규격 준수 분석
            spec_analysis = {}
            if usl is not None and lsl is not None:
                within_spec = sum(1 for v in values if lsl <= v <= usl)
                spec_yield = (within_spec / len(values)) * 100
                
                # Cp, Cpk 계산
                cp = (usl - lsl) / (6 * std_value) if std_value > 0 else float('inf')
                cpu = (usl - mean_value) / (3 * std_value) if std_value > 0 else float('inf')
                cpl = (mean_value - lsl) / (3 * std_value) if std_value > 0 else float('inf')
                cpk = min(cpu, cpl)
                
                spec_analysis = {
                    "spec_yield_percent": round(spec_yield, 2),
                    "within_spec_count": within_spec,
                    "out_of_spec_count": len(values) - within_spec,
                    "cp": round(cp, 3),
                    "cpk": round(cpk, 3),
                    "usl": usl,
                    "lsl": lsl
                }
            
            # 타겟 대비 성능
            target_analysis = {}
            if target is not None:
                bias = mean_value - target
                accuracy = abs(bias)
                target_analysis = {
                    "target_value": target,
                    "bias": round(bias, 4),
                    "accuracy": round(accuracy, 4),
                    "bias_percent": round((bias / target) * 100, 2) if target != 0 else 0
                }
            
            # 트렌드 분석
            if len(values) > 1:
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                trend_analysis = {
                    "trend_slope": round(slope, 6),
                    "trend_direction": "Increasing" if slope > 0.001 else "Decreasing" if slope < -0.001 else "Stable",
                    "trend_strength": "Strong" if abs(slope) > 0.01 else "Weak"
                }
            else:
                trend_analysis = {"trend_direction": "Insufficient data"}
            
            analysis_results[metric_name] = {
                "performance_metrics": performance_metrics,
                "specification_analysis": spec_analysis,
                "target_analysis": target_analysis,
                "trend_analysis": trend_analysis,
                "data_points": len(values)
            }
        
        # 전체 툴 성능 요약
        overall_cpks = []
        overall_yields = []
        
        for metric_analysis in analysis_results.values():
            spec_analysis = metric_analysis.get("specification_analysis", {})
            if "cpk" in spec_analysis:
                overall_cpks.append(spec_analysis["cpk"])
            if "spec_yield_percent" in spec_analysis:
                overall_yields.append(spec_analysis["spec_yield_percent"])
        
        overall_summary = {
            "total_metrics_analyzed": len(analysis_results),
            "average_cpk": round(np.mean(overall_cpks), 3) if overall_cpks else None,
            "average_yield": round(np.mean(overall_yields), 2) if overall_yields else None,
            "tool_performance_grade": "A" if (overall_cpks and np.mean(overall_cpks) >= 1.33) else 
                                    "B" if (overall_cpks and np.mean(overall_cpks) >= 1.0) else "C"
        }
        
        return {
            "tool_analysis": analysis_results,
            "overall_summary": overall_summary
        }

@mcp.tool()
def calculate_oee(planned_time: float, downtime: float, cycle_time: float, 
                 ideal_cycle_time: float, good_units: int, total_units: int) -> Dict[str, Any]:
    """
    OEE (Overall Equipment Effectiveness) 계산
    
    Args:
        planned_time: 계획 운전 시간 (시간)
        downtime: 다운타임 (시간)
        cycle_time: 실제 사이클 타임 (분/단위)
        ideal_cycle_time: 이상적 사이클 타임 (분/단위)
        good_units: 양품 수량
        total_units: 총 생산 수량
    
    Returns:
        OEE 분석 결과 (가동률, 성능률, 품질률, 손실 분석)
    """
    return EquipmentAnalyzer.calculate_oee(planned_time, downtime, cycle_time, 
                                         ideal_cycle_time, good_units, total_units)

@mcp.tool()
def calculate_mtbf_mttr(failure_times: str, repair_times: str) -> Dict[str, Any]:
    """
    MTBF/MTTR 계산
    
    Args:
        failure_times: 고장 발생 시간 리스트 (JSON 배열, ISO 형식)
        repair_times: 수리 완료 시간 리스트 (JSON 배열, ISO 형식)
    
    Returns:
        신뢰성 분석 결과 (MTBF, MTTR, 가용도)
    """
    try:
        failure_list = json.loads(failure_times)
        repair_list = json.loads(repair_times)
        return EquipmentAnalyzer.calculate_mtbf_mttr(failure_list, repair_list)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

@mcp.tool()
def analyze_equipment_utilization(production_data: str) -> Dict[str, Any]:
    """
    장비 가동률 분석
    
    Args:
        production_data: 생산 데이터 (JSON 배열)
                        각 레코드: {"duration_hours": float, "status": str, "units_produced": int}
    
    Returns:
        가동률 및 활용도 분석 결과
    """
    try:
        data_list = json.loads(production_data)
        return EquipmentAnalyzer.analyze_equipment_utilization(data_list)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

@mcp.tool()
def analyze_tool_performance(tool_metrics: str, tool_specs: str) -> Dict[str, Any]:
    """
    툴 성능 분석
    
    Args:
        tool_metrics: 툴 메트릭 데이터 (JSON 객체)
                     {"metric_name": [values], ...}
        tool_specs: 툴 규격 정보 (JSON 객체)
                   {"metric_name": {"target": float, "usl": float, "lsl": float}, ...}
    
    Returns:
        툴 성능 분석 결과 (Cp/Cpk, 수율, 트렌드)
    """
    try:
        metrics_dict = json.loads(tool_metrics)
        specs_dict = json.loads(tool_specs)
        return EquipmentAnalyzer.analyze_tool_performance(metrics_dict, specs_dict)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

@mcp.tool()
def calculate_chamber_matching(chamber_data: str) -> Dict[str, Any]:
    """
    챔버 매칭 분석 (다중 챔버 장비의 성능 일치도)
    
    Args:
        chamber_data: 챔버별 성능 데이터 (JSON 객체)
                     {"chamber_id": [performance_values], ...}
    
    Returns:
        챔버 매칭 분석 결과
    """
    try:
        data_dict = json.loads(chamber_data)
        
        if not data_dict:
            return {"error": "No chamber data provided"}
        
        chamber_stats = {}
        all_values = []
        
        # 챔버별 통계 계산
        for chamber_id, values in data_dict.items():
            if not values:
                continue
                
            chamber_mean = np.mean(values)
            chamber_std = np.std(values)
            
            chamber_stats[chamber_id] = {
                "mean": round(chamber_mean, 4),
                "std_dev": round(chamber_std, 4),
                "count": len(values),
                "min": round(np.min(values), 4),
                "max": round(np.max(values), 4)
            }
            
            all_values.extend(values)
        
        if not all_values:
            return {"error": "No valid data found"}
        
        # 전체 통계
        overall_mean = np.mean(all_values)
        overall_std = np.std(all_values)
        
        # 챔버 간 변동 분석
        chamber_means = [stats["mean"] for stats in chamber_stats.values()]
        between_chamber_std = np.std(chamber_means)
        within_chamber_stds = [stats["std_dev"] for stats in chamber_stats.values()]
        avg_within_chamber_std = np.mean(within_chamber_stds)
        
        # 매칭도 계산
        matching_ratio = avg_within_chamber_std / between_chamber_std if between_chamber_std > 0 else float('inf')
        
        # 매칭 등급
        if between_chamber_std / overall_mean < 0.01:  # 1% 이내
            matching_grade = "Excellent"
        elif between_chamber_std / overall_mean < 0.03:  # 3% 이내
            matching_grade = "Good"
        elif between_chamber_std / overall_mean < 0.05:  # 5% 이내
            matching_grade = "Fair"
        else:
            matching_grade = "Poor"
        
        # 이상 챔버 식별
        outlier_chambers = []
        for chamber_id, stats in chamber_stats.items():
            z_score = abs(stats["mean"] - overall_mean) / overall_std
            if z_score > 2:  # 2σ 이상 벗어난 챔버
                outlier_chambers.append({
                    "chamber_id": chamber_id,
                    "z_score": round(z_score, 2),
                    "deviation_percent": round(((stats["mean"] - overall_mean) / overall_mean) * 100, 2)
                })
        
        return {
            "chamber_statistics": chamber_stats,
            "overall_statistics": {
                "overall_mean": round(overall_mean, 4),
                "overall_std": round(overall_std, 4),
                "total_chambers": len(chamber_stats),
                "total_data_points": len(all_values)
            },
            "matching_analysis": {
                "between_chamber_std": round(between_chamber_std, 4),
                "within_chamber_std": round(avg_within_chamber_std, 4),
                "matching_ratio": round(matching_ratio, 3),
                "matching_grade": matching_grade,
                "uniformity_percent": round((1 - between_chamber_std / overall_mean) * 100, 2) if overall_mean > 0 else 0
            },
            "outlier_analysis": {
                "outlier_chambers": outlier_chambers,
                "outlier_count": len(outlier_chambers),
                "chambers_in_spec": len(chamber_stats) - len(outlier_chambers)
            }
        }
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

if __name__ == "__main__":
    print(f"Starting Semiconductor Equipment Analysis MCP server on port {SERVER_PORT}...")
    uvicorn.run(
        mcp,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info"
    )