#!/usr/bin/env python3
"""
MCP Tool: Process Optimization
공정 최적화 도구 - DOE, 응답 표면 방법론, 공정 파라미터 최적화
"""

import os
import uvicorn
from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import math
import json
from itertools import combinations

# 환경변수에서 포트 가져오기 (기본값: 8012)
SERVER_PORT = int(os.getenv('SERVER_PORT', '8012'))

# FastMCP 서버 생성
mcp = FastMCP("Process Optimization")

class ProcessOptimizer:
    """공정 최적화 분석기"""
    
    @staticmethod
    def design_factorial_experiment(factors: Dict[str, Dict[str, float]], 
                                   design_type: str = "full_factorial") -> Dict[str, Any]:
        """실험계획법 설계"""
        if not factors:
            return {"error": "No factors provided"}
        
        factor_names = list(factors.keys())
        factor_levels = {}
        
        # 인수별 수준 설정
        for factor_name, factor_info in factors.items():
            low = factor_info.get("low", 0)
            high = factor_info.get("high", 1)
            center = factor_info.get("center", (low + high) / 2)
            
            if design_type == "full_factorial":
                factor_levels[factor_name] = [low, high]
            elif design_type == "central_composite":
                alpha = factor_info.get("alpha", 1.414)  # 축점 거리
                factor_levels[factor_name] = [low, center, high, 
                                            center - alpha * (high - center) / 2,
                                            center + alpha * (high - center) / 2]
            else:
                factor_levels[factor_name] = [low, center, high]
        
        # 실험 조합 생성
        if design_type == "full_factorial":
            # 완전요인설계
            experiments = []
            for i in range(2 ** len(factor_names)):
                experiment = {}
                for j, factor_name in enumerate(factor_names):
                    level_index = (i >> j) & 1
                    experiment[factor_name] = factor_levels[factor_name][level_index]
                experiments.append(experiment)
        
        elif design_type == "fractional_factorial":
            # 부분요인설계 (1/2 fraction)
            experiments = []
            n_factors = len(factor_names)
            for i in range(2 ** (n_factors - 1)):
                experiment = {}
                for j, factor_name in enumerate(factor_names[:-1]):
                    level_index = (i >> j) & 1
                    experiment[factor_name] = factor_levels[factor_name][level_index]
                
                # 마지막 인수는 다른 인수들의 상호작용으로 결정
                last_factor = factor_names[-1]
                interaction_index = sum((i >> j) & 1 for j in range(n_factors - 1)) % 2
                experiment[last_factor] = factor_levels[last_factor][interaction_index]
                experiments.append(experiment)
        
        else:  # central_composite
            experiments = []
            n_factors = len(factor_names)
            
            # 요인점 (factorial points)
            for i in range(2 ** n_factors):
                experiment = {}
                for j, factor_name in enumerate(factor_names):
                    level_index = (i >> j) & 1
                    experiment[factor_name] = factor_levels[factor_name][level_index]
                experiments.append(experiment)
            
            # 중심점 (center points)
            center_experiment = {name: levels[1] for name, levels in factor_levels.items()}
            experiments.extend([center_experiment] * 3)  # 3개의 중심점
            
            # 축점 (axial points)
            for i, factor_name in enumerate(factor_names):
                for level_idx in [3, 4]:  # -α, +α
                    experiment = {name: levels[1] for name, levels in factor_levels.items()}  # 중심점 기준
                    experiment[factor_name] = factor_levels[factor_name][level_idx]
                    experiments.append(experiment)
        
        # 실험 순서 무작위화
        np.random.shuffle(experiments)
        
        # 실험 번호 추가
        for i, exp in enumerate(experiments):
            exp["run_order"] = i + 1
        
        return {
            "design_type": design_type,
            "total_runs": len(experiments),
            "factors": factor_names,
            "factor_levels": factor_levels,
            "experiment_matrix": experiments,
            "design_properties": {
                "balanced": True,
                "orthogonal": design_type in ["full_factorial", "central_composite"],
                "resolution": "V" if design_type == "full_factorial" else "III" if design_type == "fractional_factorial" else "N/A"
            }
        }
    
    @staticmethod
    def analyze_doe_results(experiment_data: List[Dict[str, Any]], response_column: str) -> Dict[str, Any]:
        """DOE 결과 분석"""
        if not experiment_data:
            return {"error": "No experiment data provided"}
        
        # 데이터프레임 생성
        df = pd.DataFrame(experiment_data)
        
        if response_column not in df.columns:
            return {"error": f"Response column '{response_column}' not found"}
        
        # 인수와 응답 분리
        factor_columns = [col for col in df.columns if col not in [response_column, "run_order"]]
        
        if not factor_columns:
            return {"error": "No factor columns found"}
        
        X = df[factor_columns].values
        y = df[response_column].values
        
        # 주효과 분석
        main_effects = {}
        for i, factor in enumerate(factor_columns):
            factor_values = X[:, i]
            unique_levels = np.unique(factor_values)
            
            if len(unique_levels) >= 2:
                # 높은 수준과 낮은 수준의 평균 응답 계산
                low_response = np.mean(y[factor_values == unique_levels[0]])
                high_response = np.mean(y[factor_values == unique_levels[-1]])
                effect = high_response - low_response
                
                main_effects[factor] = {
                    "effect": round(effect, 4),
                    "low_level_mean": round(low_response, 4),
                    "high_level_mean": round(high_response, 4),
                    "significance": "High" if abs(effect) > np.std(y) else "Low"
                }
        
        # 2인수 상호작용 분석
        interaction_effects = {}
        if len(factor_columns) >= 2:
            for i, j in combinations(range(len(factor_columns)), 2):
                factor1, factor2 = factor_columns[i], factor_columns[j]
                
                # 상호작용 효과 계산 (간단한 방법)
                f1_values = X[:, i]
                f2_values = X[:, j]
                
                f1_levels = np.unique(f1_values)
                f2_levels = np.unique(f2_values)
                
                if len(f1_levels) >= 2 and len(f2_levels) >= 2:
                    # 2x2 상호작용 계산
                    responses = {}
                    for f1_level in f1_levels[:2]:
                        for f2_level in f2_levels[:2]:
                            mask = (f1_values == f1_level) & (f2_values == f2_level)
                            if np.any(mask):
                                responses[(f1_level, f2_level)] = np.mean(y[mask])
                    
                    if len(responses) == 4:
                        # 상호작용 효과 = (AB_high - AB_low) - (A_effect + B_effect)
                        ab_high_high = responses.get((f1_levels[1], f2_levels[1]), 0)
                        ab_low_low = responses.get((f1_levels[0], f2_levels[0]), 0)
                        ab_high_low = responses.get((f1_levels[1], f2_levels[0]), 0)
                        ab_low_high = responses.get((f1_levels[0], f2_levels[1]), 0)
                        
                        interaction = ((ab_high_high + ab_low_low) - (ab_high_low + ab_low_high)) / 2
                        
                        interaction_effects[f"{factor1}*{factor2}"] = {
                            "interaction_effect": round(interaction, 4),
                            "significance": "High" if abs(interaction) > np.std(y) * 0.5 else "Low"
                        }
        
        # 모델 적합성 평가
        y_mean = np.mean(y)
        total_variation = np.sum((y - y_mean) ** 2)
        
        # 단순 선형 모델로 설명되는 변동
        explained_variation = 0
        for factor, effect_info in main_effects.items():
            effect = effect_info["effect"]
            explained_variation += (effect ** 2) * len(y) / 4  # 간단한 근사
        
        r_squared = min(explained_variation / total_variation, 1.0) if total_variation > 0 else 0
        
        # 추천사항
        recommendations = []
        
        # 가장 큰 효과를 가진 인수
        if main_effects:
            max_effect_factor = max(main_effects.keys(), key=lambda k: abs(main_effects[k]["effect"]))
            recommendations.append(f"인수 '{max_effect_factor}'가 가장 큰 영향을 미칩니다.")
        
        # 유의한 상호작용
        significant_interactions = [k for k, v in interaction_effects.items() 
                                   if v["significance"] == "High"]
        if significant_interactions:
            recommendations.append(f"유의한 상호작용: {', '.join(significant_interactions)}")
        
        return {
            "experiment_summary": {
                "total_runs": len(experiment_data),
                "factors": factor_columns,
                "response_variable": response_column,
                "response_range": [round(min(y), 4), round(max(y), 4)]
            },
            "main_effects": main_effects,
            "interaction_effects": interaction_effects,
            "model_performance": {
                "r_squared": round(r_squared, 4),
                "residual_std": round(np.std(y - y_mean), 4),
                "model_adequacy": "Good" if r_squared > 0.7 else "Fair" if r_squared > 0.5 else "Poor"
            },
            "recommendations": recommendations
        }
    
    @staticmethod
    def optimize_response_surface(factor_ranges: Dict[str, Tuple[float, float]], 
                                 response_model: Dict[str, float],
                                 optimization_goal: str = "maximize") -> Dict[str, Any]:
        """응답 표면 최적화"""
        if not factor_ranges or not response_model:
            return {"error": "Factor ranges and response model required"}
        
        factors = list(factor_ranges.keys())
        
        # 격자 탐색으로 최적점 찾기
        grid_points = 20  # 각 축당 점의 수
        best_response = float('-inf') if optimization_goal == "maximize" else float('inf')
        best_conditions = {}
        
        # 격자 생성
        factor_grids = {}
        for factor, (low, high) in factor_ranges.items():
            factor_grids[factor] = np.linspace(low, high, grid_points)
        
        # 모든 조합에 대해 응답 계산
        optimization_results = []
        
        for i in range(grid_points ** len(factors)):
            conditions = {}
            temp_i = i
            
            for factor in factors:
                grid_index = temp_i % grid_points
                conditions[factor] = factor_grids[factor][grid_index]
                temp_i //= grid_points
            
            # 응답 계산 (2차 모델 가정)
            response = response_model.get("intercept", 0)
            
            # 1차 항
            for factor in factors:
                coeff_key = factor
                if coeff_key in response_model:
                    response += response_model[coeff_key] * conditions[factor]
            
            # 2차 항
            for factor in factors:
                coeff_key = f"{factor}^2"
                if coeff_key in response_model:
                    response += response_model[coeff_key] * (conditions[factor] ** 2)
            
            # 상호작용 항
            for i, factor1 in enumerate(factors):
                for factor2 in factors[i+1:]:
                    coeff_key = f"{factor1}*{factor2}"
                    if coeff_key in response_model:
                        response += response_model[coeff_key] * conditions[factor1] * conditions[factor2]
            
            optimization_results.append({
                "conditions": conditions.copy(),
                "predicted_response": response
            })
            
            # 최적점 업데이트
            if optimization_goal == "maximize" and response > best_response:
                best_response = response
                best_conditions = conditions.copy()
            elif optimization_goal == "minimize" and response < best_response:
                best_response = response
                best_conditions = conditions.copy()
        
        # 상위 5개 조건
        sorted_results = sorted(optimization_results, 
                              key=lambda x: x["predicted_response"], 
                              reverse=(optimization_goal == "maximize"))
        top_conditions = sorted_results[:5]
        
        # 민감도 분석
        sensitivity_analysis = {}
        for factor in factors:
            # 현재 최적 조건에서 해당 인수만 변경했을 때의 응답 변화
            base_conditions = best_conditions.copy()
            factor_range = factor_ranges[factor]
            
            # ±10% 변화에 대한 응답 변화
            delta = (factor_range[1] - factor_range[0]) * 0.1
            
            low_conditions = base_conditions.copy()
            low_conditions[factor] = max(factor_range[0], base_conditions[factor] - delta)
            
            high_conditions = base_conditions.copy()
            high_conditions[factor] = min(factor_range[1], base_conditions[factor] + delta)
            
            # 응답 계산
            def calculate_response(conditions):
                response = response_model.get("intercept", 0)
                for f in factors:
                    if f in response_model:
                        response += response_model[f] * conditions[f]
                    if f"{f}^2" in response_model:
                        response += response_model[f"{f}^2"] * (conditions[f] ** 2)
                return response
            
            low_response = calculate_response(low_conditions)
            high_response = calculate_response(high_conditions)
            
            sensitivity = (high_response - low_response) / (2 * delta)
            sensitivity_analysis[factor] = {
                "sensitivity": round(sensitivity, 4),
                "impact_level": "High" if abs(sensitivity) > 1 else "Medium" if abs(sensitivity) > 0.1 else "Low"
            }
        
        return {
            "optimization_goal": optimization_goal,
            "optimal_conditions": {k: round(v, 4) for k, v in best_conditions.items()},
            "optimal_response": round(best_response, 4),
            "top_alternatives": [
                {
                    "conditions": {k: round(v, 4) for k, v in result["conditions"].items()},
                    "response": round(result["predicted_response"], 4)
                }
                for result in top_conditions
            ],
            "sensitivity_analysis": sensitivity_analysis,
            "robustness": {
                "most_sensitive_factor": max(sensitivity_analysis.keys(), 
                                           key=lambda k: abs(sensitivity_analysis[k]["sensitivity"])),
                "least_sensitive_factor": min(sensitivity_analysis.keys(), 
                                            key=lambda k: abs(sensitivity_analysis[k]["sensitivity"]))
            }
        }

@mcp.tool()
def design_factorial_experiment(factors: str, design_type: str = "full_factorial") -> Dict[str, Any]:
    """
    실험계획법 설계
    
    Args:
        factors: 인수 정의 (JSON 객체)
                {"factor_name": {"low": float, "high": float, "center": float}, ...}
        design_type: 설계 유형 ("full_factorial", "fractional_factorial", "central_composite")
    
    Returns:
        실험 매트릭스 및 설계 정보
    """
    try:
        factors_dict = json.loads(factors)
        return ProcessOptimizer.design_factorial_experiment(factors_dict, design_type)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for factors"}

@mcp.tool()
def analyze_doe_results(experiment_data: str, response_column: str) -> Dict[str, Any]:
    """
    DOE 결과 분석
    
    Args:
        experiment_data: 실험 데이터 (JSON 배열)
        response_column: 응답 변수 컬럼명
    
    Returns:
        주효과, 상호작용 효과, 모델 성능 분석
    """
    try:
        data_list = json.loads(experiment_data)
        return ProcessOptimizer.analyze_doe_results(data_list, response_column)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for experiment data"}

@mcp.tool()
def optimize_process_parameters(factor_ranges: str, current_conditions: str, 
                               target_response: float, constraint_type: str = "maximize") -> Dict[str, Any]:
    """
    공정 파라미터 최적화
    
    Args:
        factor_ranges: 인수 범위 (JSON 객체)
                      {"factor": [min, max], ...}
        current_conditions: 현재 조건 (JSON 객체)
        target_response: 목표 응답값
        constraint_type: 제약 유형 ("maximize", "minimize", "target")
    
    Returns:
        최적화 권고안
    """
    try:
        ranges_dict = json.loads(factor_ranges)
        current_dict = json.loads(current_conditions)
        
        # 현재 조건 분석
        current_analysis = {}
        improvements = []
        
        for factor, (min_val, max_val) in ranges_dict.items():
            current_val = current_dict.get(factor, (min_val + max_val) / 2)
            
            # 현재값이 범위 내에 있는지 확인
            if current_val < min_val or current_val > max_val:
                improvements.append(f"{factor}: 현재값 {current_val}이 허용 범위 [{min_val}, {max_val}]를 벗어남")
            
            # 최적화 방향 제안
            range_center = (min_val + max_val) / 2
            range_span = max_val - min_val
            
            current_analysis[factor] = {
                "current_value": current_val,
                "range": [min_val, max_val],
                "relative_position": (current_val - min_val) / range_span if range_span > 0 else 0.5,
                "suggested_direction": "increase" if current_val < range_center else "decrease" if current_val > range_center else "maintain"
            }
        
        # 단순 최적화 권고 (경험적 규칙 기반)
        optimization_recommendations = []
        
        for factor, analysis in current_analysis.items():
            pos = analysis["relative_position"]
            min_val, max_val = analysis["range"]
            
            if constraint_type == "maximize":
                if pos < 0.8:  # 상위 20% 구간이 아니면
                    new_value = min_val + 0.8 * (max_val - min_val)
                    optimization_recommendations.append({
                        "factor": factor,
                        "current": analysis["current_value"],
                        "recommended": round(new_value, 4),
                        "change": "increase",
                        "expected_impact": "positive"
                    })
            
            elif constraint_type == "minimize":
                if pos > 0.2:  # 하위 20% 구간이 아니면
                    new_value = min_val + 0.2 * (max_val - min_val)
                    optimization_recommendations.append({
                        "factor": factor,
                        "current": analysis["current_value"],
                        "recommended": round(new_value, 4),
                        "change": "decrease",
                        "expected_impact": "positive"
                    })
            
            else:  # target
                # 목표값에 가까운 조건 제안
                target_position = 0.5  # 중심값 목표
                if abs(pos - target_position) > 0.1:
                    new_value = min_val + target_position * (max_val - min_val)
                    optimization_recommendations.append({
                        "factor": factor,
                        "current": analysis["current_value"],
                        "recommended": round(new_value, 4),
                        "change": "adjust to center",
                        "expected_impact": "stabilization"
                    })
        
        # 실험 제안
        experimental_design = {
            "confirmation_runs": 3,
            "validation_conditions": [
                {factor: rec["recommended"] for rec in optimization_recommendations 
                 for factor in [rec["factor"]]}
            ],
            "control_conditions": current_dict
        }
        
        return {
            "current_analysis": current_analysis,
            "target_response": target_response,
            "optimization_goal": constraint_type,
            "recommendations": optimization_recommendations,
            "improvement_potential": len(optimization_recommendations) > 0,
            "next_steps": {
                "immediate_actions": improvements[:3],  # 상위 3개
                "experimental_plan": experimental_design,
                "monitoring_factors": [rec["factor"] for rec in optimization_recommendations]
            }
        }
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

@mcp.tool()
def analyze_process_capability_improvement(current_capability: str, target_capability: str) -> Dict[str, Any]:
    """
    공정 능력 개선 분석
    
    Args:
        current_capability: 현재 공정 능력 데이터 (JSON 객체)
                          {"cp": float, "cpk": float, "yield": float, "defect_rate": float}
        target_capability: 목표 공정 능력 (JSON 객체)
    
    Returns:
        개선 방안 및 로드맵
    """
    try:
        current = json.loads(current_capability)
        target = json.loads(target_capability)
        
        # 개선 갭 분석
        improvement_gaps = {}
        for metric in ["cp", "cpk", "yield", "defect_rate"]:
            current_val = current.get(metric, 0)
            target_val = target.get(metric, current_val)
            
            if metric == "defect_rate":
                gap = current_val - target_val  # 결함율은 감소가 목표
                improvement_needed = gap > 0
            else:
                gap = target_val - current_val  # 나머지는 증가가 목표
                improvement_needed = gap > 0
            
            improvement_gaps[metric] = {
                "current": current_val,
                "target": target_val,
                "gap": round(gap, 4),
                "improvement_needed": improvement_needed,
                "percent_improvement": round((gap / current_val) * 100, 2) if current_val > 0 else 0
            }
        
        # 우선순위 분석
        priorities = []
        if improvement_gaps["cpk"]["improvement_needed"]:
            priorities.append({
                "area": "Process Capability (Cpk)",
                "urgency": "High" if improvement_gaps["cpk"]["gap"] > 0.5 else "Medium",
                "impact": "High",
                "actions": ["Center process", "Reduce variation", "Tighten process control"]
            })
        
        if improvement_gaps["yield"]["improvement_needed"]:
            priorities.append({
                "area": "Yield Improvement",
                "urgency": "High" if improvement_gaps["yield"]["gap"] > 5 else "Medium",
                "impact": "High", 
                "actions": ["Defect reduction", "Process optimization", "Equipment maintenance"]
            })
        
        if improvement_gaps["defect_rate"]["improvement_needed"]:
            priorities.append({
                "area": "Defect Rate Reduction",
                "urgency": "Critical" if current["defect_rate"] > 1000 else "High",
                "impact": "High",
                "actions": ["Root cause analysis", "Preventive controls", "Quality systems"]
            })
        
        # 개선 로드맵
        roadmap_phases = []
        
        # Phase 1: 즉시 개선 (0-3개월)
        phase1_actions = []
        if current.get("cpk", 0) < 1.0:
            phase1_actions.extend(["공정 중심화", "관리도 구축", "즉시 조치 시스템"])
        
        if current.get("defect_rate", 0) > 1000:
            phase1_actions.append("주요 결함 원인 제거")
        
        if phase1_actions:
            roadmap_phases.append({
                "phase": "Phase 1 (0-3 months)",
                "focus": "Immediate Improvements",
                "actions": phase1_actions,
                "expected_cpk": min(current.get("cpk", 0) + 0.3, 1.0)
            })
        
        # Phase 2: 중기 개선 (3-12개월)
        phase2_actions = ["고급 SPC 구현", "DOE 기반 최적화", "예방 정비 시스템"]
        roadmap_phases.append({
            "phase": "Phase 2 (3-12 months)",
            "focus": "Systematic Optimization",
            "actions": phase2_actions,
            "expected_cpk": min(current.get("cpk", 0) + 0.6, 1.5)
        })
        
        # Phase 3: 장기 개선 (1-2년)
        phase3_actions = ["6시그마 프로젝트", "자동화 확대", "AI 기반 예측 제어"]
        roadmap_phases.append({
            "phase": "Phase 3 (1-2 years)",
            "focus": "Excellence Achievement",
            "actions": phase3_actions,
            "expected_cpk": target.get("cpk", 2.0)
        })
        
        return {
            "gap_analysis": improvement_gaps,
            "improvement_priorities": sorted(priorities, key=lambda x: x["urgency"], reverse=True),
            "improvement_roadmap": roadmap_phases,
            "investment_estimate": {
                "phase1_cost": "Low (인력 재배치 중심)",
                "phase2_cost": "Medium (시스템 구축)",
                "phase3_cost": "High (기술 투자)",
                "roi_timeline": "6-18 months"
            },
            "success_metrics": {
                "interim_milestones": [
                    {"timeline": "3 months", "cpk_target": 1.0, "defect_reduction": "50%"},
                    {"timeline": "6 months", "cpk_target": 1.2, "defect_reduction": "70%"},
                    {"timeline": "12 months", "cpk_target": 1.5, "defect_reduction": "90%"}
                ],
                "final_target": target
            }
        }
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

if __name__ == "__main__":
    print(f"Starting Process Optimization MCP server on port {SERVER_PORT}...")
    uvicorn.run(
        mcp,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info"
    )