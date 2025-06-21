#!/usr/bin/env python3
"""
MCP Tool: Semiconductor Yield Analysis
반도체 수율 분석 전문 도구 - 웨이퍼 수율, 공정 수율, 품질 분석
"""

from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import json
import os
import uvicorn

# Get port from environment variable
SERVER_PORT = int(os.getenv('SERVER_PORT', '8008'))

# FastMCP 서버 생성
mcp = FastMCP("Semiconductor Yield Analysis")

class YieldAnalyzer:
    """반도체 수율 분석기"""
    
    @staticmethod
    def calculate_wafer_yield(good_dies: int, total_dies: int, defect_density: float = None) -> Dict[str, Any]:
        """웨이퍼 수율 계산"""
        if total_dies <= 0:
            return {"error": "Total dies must be greater than 0"}
        
        yield_percent = (good_dies / total_dies) * 100
        
        result = {
            "wafer_yield_percent": round(yield_percent, 2),
            "good_dies": good_dies,
            "total_dies": total_dies,
            "defective_dies": total_dies - good_dies,
            "defect_rate_percent": round(((total_dies - good_dies) / total_dies) * 100, 2)
        }
        
        # 결함 밀도가 주어진 경우 포아송 분포 기반 이론적 수율 계산
        if defect_density is not None:
            die_area = 1.0  # 기본값, 실제로는 mm² 단위
            theoretical_yield = math.exp(-defect_density * die_area) * 100
            result["theoretical_yield_percent"] = round(theoretical_yield, 2)
            result["yield_gap_percent"] = round(theoretical_yield - yield_percent, 2)
        
        return result
    
    @staticmethod
    def analyze_process_yield(step_yields: List[float]) -> Dict[str, Any]:
        """공정별 수율 분석"""
        if not step_yields or any(y < 0 or y > 100 for y in step_yields):
            return {"error": "Invalid yield values. Must be between 0 and 100."}
        
        # 전체 수율 = 각 공정 수율의 곱
        overall_yield = np.prod([y/100 for y in step_yields]) * 100
        
        # 가장 낮은 수율 공정 식별
        min_yield_step = np.argmin(step_yields)
        
        # 수율 향상 시뮬레이션
        improvement_scenarios = []
        for i, yield_val in enumerate(step_yields):
            improved_yields = step_yields.copy()
            improved_yields[i] = min(yield_val + 5, 100)  # 5% 향상
            improved_overall = np.prod([y/100 for y in improved_yields]) * 100
            
            improvement_scenarios.append({
                "step": i,
                "current_yield": yield_val,
                "improved_yield": improved_yields[i],
                "overall_impact": round(improved_overall - overall_yield, 2)
            })
        
        return {
            "overall_yield_percent": round(overall_yield, 2),
            "step_yields": step_yields,
            "bottleneck_step": min_yield_step,
            "bottleneck_yield": step_yields[min_yield_step],
            "improvement_scenarios": improvement_scenarios,
            "total_steps": len(step_yields)
        }
    
    @staticmethod
    def calculate_dpmo(defects: int, units: int, opportunities: int) -> Dict[str, Any]:
        """DPMO (Defects Per Million Opportunities) 계산"""
        if units <= 0 or opportunities <= 0:
            return {"error": "Units and opportunities must be greater than 0"}
        
        dpmo = (defects / (units * opportunities)) * 1_000_000
        
        # 시그마 레벨 계산 (근사치)
        if dpmo <= 0:
            sigma_level = 6.0
        else:
            # 정규분포 역함수 근사
            yield_rate = 1 - (dpmo / 1_000_000)
            if yield_rate >= 0.9999966:  # 6시그마 수준
                sigma_level = 6.0
            elif yield_rate >= 0.999767:  # 5시그마 수준
                sigma_level = 5.0 + (yield_rate - 0.999767) / (0.9999966 - 0.999767)
            elif yield_rate >= 0.99379:   # 4시그마 수준
                sigma_level = 4.0 + (yield_rate - 0.99379) / (0.999767 - 0.99379)
            elif yield_rate >= 0.93319:   # 3시그마 수준
                sigma_level = 3.0 + (yield_rate - 0.93319) / (0.99379 - 0.93319)
            else:
                sigma_level = 3.0 * yield_rate / 0.93319
        
        return {
            "dpmo": round(dpmo, 2),
            "sigma_level": round(sigma_level, 2),
            "yield_percent": round((1 - dpmo/1_000_000) * 100, 4),
            "defects": defects,
            "units": units,
            "opportunities": opportunities,
            "quality_level": "Excellent" if sigma_level >= 5 else "Good" if sigma_level >= 4 else "Average" if sigma_level >= 3 else "Poor"
        }
    
    @staticmethod
    def analyze_bin_map(bin_data: List[List[int]], bin_descriptions: Dict[int, str] = None) -> Dict[str, Any]:
        """빈 맵 분석 (웨이퍼 테스트 결과)"""
        if not bin_data or not all(isinstance(row, list) for row in bin_data):
            return {"error": "Invalid bin data format"}
        
        # 빈 맵을 numpy 배열로 변환
        bin_array = np.array(bin_data)
        total_dies = bin_array.size
        
        # 빈별 카운트
        unique_bins, counts = np.unique(bin_array, return_counts=True)
        bin_counts = dict(zip(unique_bins.tolist(), counts.tolist()))
        
        # 기본 빈 설명 (1=Pass, 0=Fail 등)
        default_descriptions = {1: "Pass", 0: "Fail"}
        descriptions = bin_descriptions or default_descriptions
        
        # 빈별 분석
        bin_analysis = []
        for bin_num, count in bin_counts.items():
            percentage = (count / total_dies) * 100
            bin_analysis.append({
                "bin": bin_num,
                "count": count,
                "percentage": round(percentage, 2),
                "description": descriptions.get(bin_num, f"Bin {bin_num}")
            })
        
        # 패스 빈 계산 (일반적으로 빈 1)
        pass_count = bin_counts.get(1, 0)
        yield_percent = (pass_count / total_dies) * 100
        
        # 공간적 클러스터링 분석 (간단한 인접성 기반)
        def find_clusters(bin_array, target_bin):
            """특정 빈의 클러스터 찾기"""
            clusters = []
            visited = np.zeros_like(bin_array, dtype=bool)
            
            def dfs(r, c, cluster):
                if (r < 0 or r >= bin_array.shape[0] or 
                    c < 0 or c >= bin_array.shape[1] or 
                    visited[r, c] or bin_array[r, c] != target_bin):
                    return
                
                visited[r, c] = True
                cluster.append((r, c))
                
                # 8방향 탐색
                for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    dfs(r + dr, c + dc, cluster)
            
            for r in range(bin_array.shape[0]):
                for c in range(bin_array.shape[1]):
                    if not visited[r, c] and bin_array[r, c] == target_bin:
                        cluster = []
                        dfs(r, c, cluster)
                        if cluster:
                            clusters.append(cluster)
            
            return clusters
        
        # 불량 클러스터 분석 (빈 0)
        fail_clusters = find_clusters(bin_array, 0)
        cluster_analysis = {
            "total_clusters": len(fail_clusters),
            "largest_cluster_size": max(len(cluster) for cluster in fail_clusters) if fail_clusters else 0,
            "cluster_sizes": [len(cluster) for cluster in fail_clusters]
        }
        
        return {
            "total_dies": total_dies,
            "yield_percent": round(yield_percent, 2),
            "bin_analysis": sorted(bin_analysis, key=lambda x: x["count"], reverse=True),
            "cluster_analysis": cluster_analysis,
            "wafer_map_size": f"{bin_array.shape[0]}x{bin_array.shape[1]}",
            "most_common_fail_bin": max((b for b in bin_counts.keys() if b != 1), 
                                      key=lambda x: bin_counts[x], default=None)
        }

@mcp.tool()
def calculate_wafer_yield(good_dies: int, total_dies: int, defect_density: float = None) -> Dict[str, Any]:
    """
    웨이퍼 수율 계산
    
    Args:
        good_dies: 양품 다이 수
        total_dies: 전체 다이 수  
        defect_density: 결함 밀도 (선택사항)
    
    Returns:
        수율 분석 결과
    """
    return YieldAnalyzer.calculate_wafer_yield(good_dies, total_dies, defect_density)

@mcp.tool()
def analyze_process_yield(step_yields: List[float]) -> Dict[str, Any]:
    """
    공정별 수율 분석
    
    Args:
        step_yields: 각 공정 단계의 수율 리스트 (퍼센트)
    
    Returns:
        공정 수율 분석 결과 및 개선 시나리오
    """
    return YieldAnalyzer.analyze_process_yield(step_yields)

@mcp.tool()
def calculate_dpmo(defects: int, units: int, opportunities: int) -> Dict[str, Any]:
    """
    DPMO 및 시그마 레벨 계산
    
    Args:
        defects: 결함 수
        units: 검사 단위 수
        opportunities: 단위당 결함 발생 기회 수
    
    Returns:
        DPMO, 시그마 레벨, 품질 등급
    """
    return YieldAnalyzer.calculate_dpmo(defects, units, opportunities)

@mcp.tool()
def analyze_bin_map(bin_data: str, bin_descriptions: str = None) -> Dict[str, Any]:
    """
    웨이퍼 빈 맵 분석
    
    Args:
        bin_data: 빈 맵 데이터 (JSON 문자열로 된 2D 배열)
        bin_descriptions: 빈 설명 (JSON 문자열, 선택사항)
    
    Returns:
        빈 맵 분석 결과, 클러스터 정보
    """
    try:
        bin_array = json.loads(bin_data)
        descriptions = json.loads(bin_descriptions) if bin_descriptions else None
        return YieldAnalyzer.analyze_bin_map(bin_array, descriptions)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for bin data"}

@mcp.tool()
def calculate_yield_pareto(defect_categories: Dict[str, int]) -> Dict[str, Any]:
    """
    수율 손실 파레토 분석
    
    Args:
        defect_categories: 결함 카테고리별 발생 수 (딕셔너리)
    
    Returns:
        파레토 분석 결과
    """
    if not defect_categories:
        return {"error": "No defect categories provided"}
    
    # 결함 수 기준 정렬
    sorted_defects = sorted(defect_categories.items(), key=lambda x: x[1], reverse=True)
    total_defects = sum(defect_categories.values())
    
    pareto_analysis = []
    cumulative_percent = 0
    
    for category, count in sorted_defects:
        percentage = (count / total_defects) * 100
        cumulative_percent += percentage
        
        pareto_analysis.append({
            "category": category,
            "count": count,
            "percentage": round(percentage, 2),
            "cumulative_percentage": round(cumulative_percent, 2),
            "is_vital_few": cumulative_percent <= 80  # 80/20 법칙
        })
    
    vital_few = [item for item in pareto_analysis if item["is_vital_few"]]
    
    return {
        "total_defects": total_defects,
        "pareto_analysis": pareto_analysis,
        "vital_few_categories": len(vital_few),
        "vital_few_impact": round(sum(item["percentage"] for item in vital_few), 2),
        "recommendation": f"상위 {len(vital_few)}개 카테고리가 전체 결함의 {round(sum(item['percentage'] for item in vital_few), 1)}%를 차지합니다."
    }

@mcp.tool()
def predict_yield_trend(historical_yields: List[float], periods_ahead: int = 5) -> Dict[str, Any]:
    """
    수율 트렌드 예측 (단순 이동평균 및 선형 회귀)
    
    Args:
        historical_yields: 과거 수율 데이터 (시간순)
        periods_ahead: 예측할 기간 수
    
    Returns:
        수율 트렌드 예측 결과
    """
    if len(historical_yields) < 3:
        return {"error": "At least 3 historical data points required"}
    
    yields = np.array(historical_yields)
    n = len(yields)
    
    # 단순 이동평균 (3기간)
    if n >= 3:
        ma3 = np.convolve(yields, np.ones(3)/3, mode='valid')
        ma3_last = ma3[-1]
    else:
        ma3_last = np.mean(yields)
    
    # 선형 회귀 트렌드
    x = np.arange(n)
    slope, intercept = np.polyfit(x, yields, 1)
    
    # 예측
    future_x = np.arange(n, n + periods_ahead)
    linear_predictions = slope * future_x + intercept
    
    # 이동평균 기반 예측 (단순하게 마지막 MA 값 유지)
    ma_predictions = [ma3_last] * periods_ahead
    
    # 트렌드 분석
    if slope > 0.1:
        trend = "Improving"
    elif slope < -0.1:
        trend = "Declining"
    else:
        trend = "Stable"
    
    # 변동성 계산
    volatility = np.std(yields)
    
    return {
        "historical_data": historical_yields,
        "trend": trend,
        "slope": round(slope, 4),
        "volatility": round(volatility, 2),
        "current_ma3": round(ma3_last, 2),
        "linear_predictions": [round(p, 2) for p in linear_predictions],
        "ma_predictions": [round(p, 2) for p in ma_predictions],
        "confidence": "High" if volatility < 2 else "Medium" if volatility < 5 else "Low",
        "recommendation": f"트렌드: {trend}, 변동성: {'낮음' if volatility < 2 else '보통' if volatility < 5 else '높음'}"
    }

if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Semiconductor Yield Analysis MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)