#!/usr/bin/env python3
"""
MCP Tool: Defect Pattern Analysis
결함 패턴 분석 도구 - 웨이퍼 맵 분석, 결함 분류, 공간 분석
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
from collections import Counter

# 환경변수에서 포트 가져오기 (기본값: 8011)
SERVER_PORT = int(os.getenv('SERVER_PORT', '8011'))

# FastMCP 서버 생성
mcp = FastMCP("Defect Pattern Analysis")

class DefectPatternAnalyzer:
    """결함 패턴 분석기"""
    
    @staticmethod
    def analyze_wafer_map_patterns(wafer_map: List[List[int]], die_size: Tuple[float, float] = (10, 10)) -> Dict[str, Any]:
        """웨이퍼 맵 패턴 분석"""
        if not wafer_map or not all(isinstance(row, list) for row in wafer_map):
            return {"error": "Invalid wafer map format"}
        
        wafer_array = np.array(wafer_map)
        rows, cols = wafer_array.shape
        total_dies = rows * cols
        
        # 기본 통계
        unique_values, counts = np.unique(wafer_array, return_counts=True)
        die_counts = dict(zip(unique_values.tolist(), counts.tolist()))
        
        # Pass/Fail 분석 (1=Pass, 0=Fail 가정)
        pass_count = die_counts.get(1, 0)
        fail_count = total_dies - pass_count
        yield_percent = (pass_count / total_dies) * 100
        
        # 공간 패턴 분석
        patterns = DefectPatternAnalyzer._detect_spatial_patterns(wafer_array)
        
        # 결함 클러스터 분석
        clusters = DefectPatternAnalyzer._find_defect_clusters(wafer_array)
        
        # 웨이퍼 영역별 분석
        region_analysis = DefectPatternAnalyzer._analyze_wafer_regions(wafer_array)
        
        # 결함 밀도 분석
        defect_density = DefectPatternAnalyzer._calculate_defect_density(wafer_array, die_size)
        
        return {
            "wafer_statistics": {
                "total_dies": total_dies,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "yield_percent": round(yield_percent, 2),
                "die_distribution": die_counts,
                "wafer_size": f"{rows}x{cols}"
            },
            "spatial_patterns": patterns,
            "cluster_analysis": clusters,
            "region_analysis": region_analysis,
            "defect_density": defect_density
        }
    
    @staticmethod
    def _detect_spatial_patterns(wafer_array: np.ndarray) -> Dict[str, Any]:
        """공간 패턴 탐지"""
        rows, cols = wafer_array.shape
        patterns = []
        
        # 엣지 효과 (Edge effect)
        edge_width = min(3, min(rows, cols) // 4)
        edge_mask = np.zeros_like(wafer_array, dtype=bool)
        edge_mask[:edge_width, :] = True
        edge_mask[-edge_width:, :] = True
        edge_mask[:, :edge_width] = True
        edge_mask[:, -edge_width:] = True
        
        edge_fail_rate = np.sum((wafer_array == 0) & edge_mask) / np.sum(edge_mask) if np.sum(edge_mask) > 0 else 0
        center_fail_rate = np.sum((wafer_array == 0) & ~edge_mask) / np.sum(~edge_mask) if np.sum(~edge_mask) > 0 else 0
        
        if edge_fail_rate > center_fail_rate * 1.5:
            patterns.append({
                "type": "Edge Effect",
                "severity": "High" if edge_fail_rate > center_fail_rate * 2 else "Medium",
                "edge_fail_rate": round(edge_fail_rate * 100, 2),
                "center_fail_rate": round(center_fail_rate * 100, 2)
            })
        
        # 센터 결함 (Center defect)
        center_r = min(rows, cols) // 4
        center_y, center_x = rows // 2, cols // 2
        center_mask = np.zeros_like(wafer_array, dtype=bool)
        
        for i in range(rows):
            for j in range(cols):
                if (i - center_y)**2 + (j - center_x)**2 <= center_r**2:
                    center_mask[i, j] = True
        
        center_core_fail_rate = np.sum((wafer_array == 0) & center_mask) / np.sum(center_mask) if np.sum(center_mask) > 0 else 0
        outer_fail_rate = np.sum((wafer_array == 0) & ~center_mask) / np.sum(~center_mask) if np.sum(~center_mask) > 0 else 0
        
        if center_core_fail_rate > outer_fail_rate * 1.5:
            patterns.append({
                "type": "Center Defect",
                "severity": "High" if center_core_fail_rate > outer_fail_rate * 2 else "Medium",
                "center_fail_rate": round(center_core_fail_rate * 100, 2),
                "outer_fail_rate": round(outer_fail_rate * 100, 2)
            })
        
        # 링 패턴 (Ring pattern)
        ring_patterns = DefectPatternAnalyzer._detect_ring_patterns(wafer_array)
        patterns.extend(ring_patterns)
        
        # 선형 패턴 (Linear pattern)
        linear_patterns = DefectPatternAnalyzer._detect_linear_patterns(wafer_array)
        patterns.extend(linear_patterns)
        
        return {
            "detected_patterns": patterns,
            "pattern_count": len(patterns),
            "dominant_pattern": patterns[0]["type"] if patterns else "Random"
        }
    
    @staticmethod
    def _detect_ring_patterns(wafer_array: np.ndarray) -> List[Dict[str, Any]]:
        """링 패턴 탐지"""
        rows, cols = wafer_array.shape
        center_y, center_x = rows // 2, cols // 2
        max_radius = min(rows, cols) // 2
        
        patterns = []
        ring_width = max(2, max_radius // 10)
        
        for r in range(ring_width, max_radius, ring_width):
            ring_mask = np.zeros_like(wafer_array, dtype=bool)
            
            for i in range(rows):
                for j in range(cols):
                    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    if r - ring_width <= distance < r:
                        ring_mask[i, j] = True
            
            if np.sum(ring_mask) == 0:
                continue
            
            ring_fail_rate = np.sum((wafer_array == 0) & ring_mask) / np.sum(ring_mask)
            overall_fail_rate = np.sum(wafer_array == 0) / wafer_array.size
            
            if ring_fail_rate > overall_fail_rate * 1.8:
                patterns.append({
                    "type": "Ring Pattern",
                    "radius": r,
                    "severity": "High" if ring_fail_rate > overall_fail_rate * 2.5 else "Medium",
                    "ring_fail_rate": round(ring_fail_rate * 100, 2),
                    "overall_fail_rate": round(overall_fail_rate * 100, 2)
                })
        
        return patterns
    
    @staticmethod
    def _detect_linear_patterns(wafer_array: np.ndarray) -> List[Dict[str, Any]]:
        """선형 패턴 탐지"""
        patterns = []
        
        # 수평/수직 라인 체크
        for axis, axis_name in [(0, "Horizontal"), (1, "Vertical")]:
            line_fail_rates = []
            
            for i in range(wafer_array.shape[axis]):
                if axis == 0:  # 수평 라인
                    line = wafer_array[i, :]
                else:  # 수직 라인
                    line = wafer_array[:, i]
                
                fail_rate = np.sum(line == 0) / len(line) if len(line) > 0 else 0
                line_fail_rates.append(fail_rate)
            
            overall_fail_rate = np.sum(wafer_array == 0) / wafer_array.size
            
            # 임계치를 넘는 라인 찾기
            for i, fail_rate in enumerate(line_fail_rates):
                if fail_rate > overall_fail_rate * 2 and fail_rate > 0.3:
                    patterns.append({
                        "type": f"{axis_name} Line Pattern",
                        "line_index": i,
                        "severity": "High" if fail_rate > 0.7 else "Medium",
                        "line_fail_rate": round(fail_rate * 100, 2),
                        "overall_fail_rate": round(overall_fail_rate * 100, 2)
                    })
        
        return patterns
    
    @staticmethod
    def _find_defect_clusters(wafer_array: np.ndarray) -> Dict[str, Any]:
        """결함 클러스터 분석"""
        # 연결된 결함 영역 찾기 (8-connectivity)
        def flood_fill(arr, start_pos, visited):
            rows, cols = arr.shape
            stack = [start_pos]
            cluster = []
            
            while stack:
                r, c = stack.pop()
                if (r < 0 or r >= rows or c < 0 or c >= cols or 
                    visited[r, c] or arr[r, c] != 0):
                    continue
                
                visited[r, c] = True
                cluster.append((r, c))
                
                # 8방향 탐색
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        stack.append((r + dr, c + dc))
            
            return cluster
        
        visited = np.zeros_like(wafer_array, dtype=bool)
        clusters = []
        
        rows, cols = wafer_array.shape
        for r in range(rows):
            for c in range(cols):
                if not visited[r, c] and wafer_array[r, c] == 0:
                    cluster = flood_fill(wafer_array, (r, c), visited)
                    if len(cluster) > 1:  # 단일 결함은 제외
                        clusters.append(cluster)
        
        # 클러스터 분석
        if not clusters:
            return {
                "cluster_count": 0,
                "largest_cluster_size": 0,
                "cluster_coverage_percent": 0,
                "cluster_analysis": []
            }
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        largest_cluster = max(clusters, key=len)
        total_clustered_defects = sum(cluster_sizes)
        total_defects = np.sum(wafer_array == 0)
        
        cluster_analysis = []
        for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)[:5]):  # 상위 5개
            # 클러스터 중심점
            center_r = np.mean([pos[0] for pos in cluster])
            center_c = np.mean([pos[1] for pos in cluster])
            
            # 클러스터 밀도
            min_r, max_r = min(pos[0] for pos in cluster), max(pos[0] for pos in cluster)
            min_c, max_c = min(pos[1] for pos in cluster), max(pos[1] for pos in cluster)
            bounding_area = (max_r - min_r + 1) * (max_c - min_c + 1)
            density = len(cluster) / bounding_area if bounding_area > 0 else 0
            
            cluster_analysis.append({
                "rank": i + 1,
                "size": len(cluster),
                "center": (round(center_r, 1), round(center_c, 1)),
                "density": round(density, 3),
                "bounding_box": f"({min_r},{min_c}) to ({max_r},{max_c})"
            })
        
        return {
            "cluster_count": len(clusters),
            "largest_cluster_size": len(largest_cluster),
            "cluster_coverage_percent": round((total_clustered_defects / total_defects) * 100, 2) if total_defects > 0 else 0,
            "average_cluster_size": round(np.mean(cluster_sizes), 2),
            "cluster_analysis": cluster_analysis,
            "clustering_tendency": "High" if len(clusters) > 5 and np.mean(cluster_sizes) > 3 else "Low"
        }
    
    @staticmethod
    def _analyze_wafer_regions(wafer_array: np.ndarray) -> Dict[str, Any]:
        """웨이퍼 영역별 분석"""
        rows, cols = wafer_array.shape
        
        # 웨이퍼를 9개 영역으로 분할 (3x3 그리드)
        region_rows = [rows // 3, rows // 3, rows - 2 * (rows // 3)]
        region_cols = [cols // 3, cols // 3, cols - 2 * (cols // 3)]
        
        regions = {}
        region_names = [
            ["Top-Left", "Top-Center", "Top-Right"],
            ["Mid-Left", "Center", "Mid-Right"],
            ["Bottom-Left", "Bottom-Center", "Bottom-Right"]
        ]
        
        r_start = 0
        for i in range(3):
            c_start = 0
            for j in range(3):
                r_end = r_start + region_rows[i]
                c_end = c_start + region_cols[j]
                
                region_data = wafer_array[r_start:r_end, c_start:c_end]
                region_name = region_names[i][j]
                
                total_dies = region_data.size
                fail_count = np.sum(region_data == 0)
                fail_rate = (fail_count / total_dies) * 100 if total_dies > 0 else 0
                
                regions[region_name] = {
                    "total_dies": total_dies,
                    "fail_count": fail_count,
                    "fail_rate_percent": round(fail_rate, 2),
                    "coordinates": f"({r_start},{c_start}) to ({r_end-1},{c_end-1})"
                }
                
                c_start = c_end
            r_start = r_end
        
        # 영역별 비교
        fail_rates = [region["fail_rate_percent"] for region in regions.values()]
        worst_region = max(regions.keys(), key=lambda k: regions[k]["fail_rate_percent"])
        best_region = min(regions.keys(), key=lambda k: regions[k]["fail_rate_percent"])
        
        return {
            "region_details": regions,
            "worst_region": {
                "name": worst_region,
                "fail_rate": regions[worst_region]["fail_rate_percent"]
            },
            "best_region": {
                "name": best_region,
                "fail_rate": regions[best_region]["fail_rate_percent"]
            },
            "region_uniformity": {
                "std_dev": round(np.std(fail_rates), 2),
                "range": round(max(fail_rates) - min(fail_rates), 2),
                "cv_percent": round((np.std(fail_rates) / np.mean(fail_rates)) * 100, 2) if np.mean(fail_rates) > 0 else 0
            }
        }
    
    @staticmethod
    def _calculate_defect_density(wafer_array: np.ndarray, die_size: Tuple[float, float]) -> Dict[str, Any]:
        """결함 밀도 계산"""
        rows, cols = wafer_array.shape
        die_width, die_height = die_size
        
        # 웨이퍼 면적 계산 (mm²)
        wafer_area_mm2 = rows * cols * die_width * die_height
        
        # 결함 수
        total_defects = np.sum(wafer_array == 0)
        
        # 결함 밀도 (defects/cm²)
        defect_density_per_cm2 = total_defects / (wafer_area_mm2 / 100) if wafer_area_mm2 > 0 else 0
        
        # 포아송 분포 기반 예상 수율
        expected_yield = math.exp(-defect_density_per_cm2 * (die_width * die_height / 100)) * 100
        
        return {
            "defect_density_per_cm2": round(defect_density_per_cm2, 4),
            "total_defects": total_defects,
            "wafer_area_mm2": round(wafer_area_mm2, 2),
            "die_size_mm": {"width": die_width, "height": die_height},
            "expected_yield_percent": round(expected_yield, 2),
            "defect_density_classification": (
                "Very Low" if defect_density_per_cm2 < 0.1 else
                "Low" if defect_density_per_cm2 < 0.5 else
                "Medium" if defect_density_per_cm2 < 1.0 else
                "High" if defect_density_per_cm2 < 2.0 else
                "Very High"
            )
        }

@mcp.tool()
def analyze_wafer_map_patterns(wafer_map: str, die_width: float = 10.0, die_height: float = 10.0) -> Dict[str, Any]:
    """
    웨이퍼 맵 패턴 분석
    
    Args:
        wafer_map: 웨이퍼 맵 데이터 (JSON 2D 배열, 1=Pass, 0=Fail)
        die_width: 다이 너비 (mm)
        die_height: 다이 높이 (mm)
    
    Returns:
        웨이퍼 맵 패턴 분석 결과
    """
    try:
        map_data = json.loads(wafer_map)
        return DefectPatternAnalyzer.analyze_wafer_map_patterns(map_data, (die_width, die_height))
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for wafer map"}

@mcp.tool()
def classify_defect_signatures(defect_data: str) -> Dict[str, Any]:
    """
    결함 시그니처 분류
    
    Args:
        defect_data: 결함 데이터 (JSON 배열)
                    각 결함: {"type": str, "size": float, "location": [x, y], "severity": str}
    
    Returns:
        결함 분류 및 통계 분석
    """
    try:
        defects = json.loads(defect_data)
        
        if not defects:
            return {"error": "No defect data provided"}
        
        # 결함 타입별 분류
        defect_types = Counter(defect.get("type", "Unknown") for defect in defects)
        
        # 크기별 분석
        sizes = [defect.get("size", 0) for defect in defects if defect.get("size")]
        size_stats = {
            "mean_size": round(np.mean(sizes), 2) if sizes else 0,
            "std_size": round(np.std(sizes), 2) if sizes else 0,
            "min_size": round(min(sizes), 2) if sizes else 0,
            "max_size": round(max(sizes), 2) if sizes else 0
        }
        
        # 심각도별 분류
        severity_counts = Counter(defect.get("severity", "Unknown") for defect in defects)
        
        # 공간 분포 분석
        locations = [defect.get("location", [0, 0]) for defect in defects if defect.get("location")]
        if locations:
            x_coords = [loc[0] for loc in locations]
            y_coords = [loc[1] for loc in locations]
            
            spatial_stats = {
                "center_of_mass": [round(np.mean(x_coords), 2), round(np.mean(y_coords), 2)],
                "spread_x": round(np.std(x_coords), 2),
                "spread_y": round(np.std(y_coords), 2),
                "bounding_box": {
                    "min_x": min(x_coords), "max_x": max(x_coords),
                    "min_y": min(y_coords), "max_y": max(y_coords)
                }
            }
        else:
            spatial_stats = {"error": "No location data available"}
        
        # 파레토 분석
        total_defects = len(defects)
        sorted_types = sorted(defect_types.items(), key=lambda x: x[1], reverse=True)
        
        pareto_analysis = []
        cumulative_percent = 0
        for defect_type, count in sorted_types:
            percent = (count / total_defects) * 100
            cumulative_percent += percent
            pareto_analysis.append({
                "type": defect_type,
                "count": count,
                "percent": round(percent, 2),
                "cumulative_percent": round(cumulative_percent, 2)
            })
        
        # 상위 80% 차지하는 결함 타입
        vital_few = [item for item in pareto_analysis if item["cumulative_percent"] <= 80]
        
        return {
            "total_defects": total_defects,
            "defect_classification": {
                "by_type": dict(defect_types),
                "by_severity": dict(severity_counts),
                "unique_types": len(defect_types)
            },
            "size_analysis": size_stats,
            "spatial_distribution": spatial_stats,
            "pareto_analysis": {
                "detailed": pareto_analysis,
                "vital_few_types": len(vital_few),
                "vital_few_impact": round(sum(item["percent"] for item in vital_few), 2)
            }
        }
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for defect data"}

@mcp.tool()
def analyze_lot_level_patterns(lot_data: str) -> Dict[str, Any]:
    """
    로트 레벨 패턴 분석
    
    Args:
        lot_data: 로트별 데이터 (JSON 배열)
                 각 로트: {"lot_id": str, "yield": float, "defect_count": int, "process_conditions": dict}
    
    Returns:
        로트 간 패턴 및 트렌드 분석
    """
    try:
        lots = json.loads(lot_data)
        
        if not lots:
            return {"error": "No lot data provided"}
        
        # 기본 통계
        yields = [lot.get("yield", 0) for lot in lots]
        defect_counts = [lot.get("defect_count", 0) for lot in lots]
        
        yield_stats = {
            "mean_yield": round(np.mean(yields), 2),
            "std_yield": round(np.std(yields), 2),
            "min_yield": round(min(yields), 2),
            "max_yield": round(max(yields), 2),
            "yield_range": round(max(yields) - min(yields), 2)
        }
        
        # 트렌드 분석
        if len(yields) > 1:
            x = np.arange(len(yields))
            slope, intercept = np.polyfit(x, yields, 1)
            
            trend_analysis = {
                "slope": round(slope, 4),
                "trend_direction": "Improving" if slope > 0.1 else "Declining" if slope < -0.1 else "Stable",
                "r_squared": round(np.corrcoef(x, yields)[0, 1]**2, 3) if len(yields) > 2 else 0
            }
        else:
            trend_analysis = {"trend_direction": "Insufficient data"}
        
        # 아웃라이어 식별
        yield_mean = np.mean(yields)
        yield_std = np.std(yields)
        outliers = []
        
        for i, lot in enumerate(lots):
            yield_val = lot.get("yield", 0)
            z_score = abs(yield_val - yield_mean) / yield_std if yield_std > 0 else 0
            
            if z_score > 2:  # 2σ 이상
                outliers.append({
                    "lot_id": lot.get("lot_id", f"Lot_{i}"),
                    "yield": yield_val,
                    "z_score": round(z_score, 2),
                    "deviation_type": "Low" if yield_val < yield_mean else "High"
                })
        
        # 공정 조건 분석 (가능한 경우)
        process_analysis = {}
        if lots and lots[0].get("process_conditions"):
            condition_keys = set()
            for lot in lots:
                conditions = lot.get("process_conditions", {})
                condition_keys.update(conditions.keys())
            
            for key in condition_keys:
                values = [lot.get("process_conditions", {}).get(key) for lot in lots]
                values = [v for v in values if v is not None]
                
                if values and all(isinstance(v, (int, float)) for v in values):
                    # 수치형 조건과 수율 간 상관관계
                    correlation = np.corrcoef(values, yields[:len(values)])[0, 1] if len(values) > 1 else 0
                    
                    process_analysis[key] = {
                        "correlation_with_yield": round(correlation, 3),
                        "mean_value": round(np.mean(values), 3),
                        "std_value": round(np.std(values), 3)
                    }
        
        return {
            "lot_statistics": {
                "total_lots": len(lots),
                "yield_statistics": yield_stats,
                "defect_statistics": {
                    "mean_defects": round(np.mean(defect_counts), 2),
                    "total_defects": sum(defect_counts)
                }
            },
            "trend_analysis": trend_analysis,
            "outlier_analysis": {
                "outlier_count": len(outliers),
                "outliers": outliers
            },
            "process_correlation": process_analysis,
            "recommendations": {
                "stability": "Good" if yield_stats["std_yield"] < 2 else "Poor",
                "investigation_needed": len(outliers) > 0,
                "trend_concern": trend_analysis.get("trend_direction") == "Declining"
            }
        }
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for lot data"}

if __name__ == "__main__":
    print(f"Starting Defect Pattern Analysis MCP server on port {SERVER_PORT}...")
    uvicorn.run(
        mcp,
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info"
    )