#!/usr/bin/env python3
"""
MCP Tool: Semiconductor Process Tools
반도체 공정 분석 도구 - 수율 분석, 웨이퍼 맵, 공정 최적화, SPC, 근본원인 분석
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
SERVER_PORT = int(os.getenv('SERVER_PORT', '8020'))

# FastMCP 서버 생성
mcp = FastMCP("Semiconductor Process Tools")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiconductorProcessTools:
    """반도체 공정 분석 도구 클래스"""
    
    @staticmethod
    def analyze_yield_performance(data: List[List[Union[float, str]]], 
                                column_names: List[str] = None,
                                analysis_scope: str = "lot",
                                target_yield: float = 0.95) -> Dict[str, Any]:
        """수율 성능 분석"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            if "yield" not in df.columns:
                return {"error": "수율 데이터(yield 컬럼)가 필요합니다."}
            
            # 기본 수율 통계
            yield_stats = {
                "overall_yield": float(df["yield"].mean()),
                "yield_std": float(df["yield"].std()),
                "yield_min": float(df["yield"].min()),
                "yield_max": float(df["yield"].max()),
                "sample_count": len(df),
                "target_yield": target_yield,
                "yield_gap": float(target_yield - df["yield"].mean())
            }
            
            # 수율 등급 분류
            overall_yield = yield_stats["overall_yield"]
            if overall_yield >= 0.95:
                grade = "Excellent"
                color = "green"
            elif overall_yield >= 0.90:
                grade = "Good" 
                color = "blue"
            elif overall_yield >= 0.80:
                grade = "Fair"
                color = "yellow"
            elif overall_yield >= 0.70:
                grade = "Poor"
                color = "orange"
            else:
                grade = "Critical"
                color = "red"
            
            yield_stats["performance_grade"] = {
                "grade": grade,
                "color": color,
                "meets_target": overall_yield >= target_yield
            }
            
            # 범위별 수율 분석
            if analysis_scope == "lot" and "lot_id" in df.columns:
                lot_analysis = SemiconductorProcessTools._analyze_by_lot(df)
                yield_stats["lot_analysis"] = lot_analysis
            
            elif analysis_scope == "wafer" and "wafer_id" in df.columns:
                wafer_analysis = SemiconductorProcessTools._analyze_by_wafer(df)
                yield_stats["wafer_analysis"] = wafer_analysis
            
            elif analysis_scope == "equipment" and "equipment_id" in df.columns:
                equipment_analysis = SemiconductorProcessTools._analyze_by_equipment(df)
                yield_stats["equipment_analysis"] = equipment_analysis
            
            # 수율 분포 분석
            yield_distribution = SemiconductorProcessTools._analyze_yield_distribution(df["yield"])
            yield_stats["yield_distribution"] = yield_distribution
            
            # 트렌드 분석 (타임스탬프가 있는 경우)
            if "timestamp" in df.columns:
                trend_analysis = SemiconductorProcessTools._analyze_yield_trend(df)
                yield_stats["trend_analysis"] = trend_analysis
            
            return {
                "yield_performance": yield_stats,
                "analysis_scope": analysis_scope,
                "recommendations": SemiconductorProcessTools._generate_yield_recommendations(yield_stats),
                "interpretation": f"전체 수율 {overall_yield:.1%} ({grade}), 목표 대비 {yield_stats['yield_gap']:+.1%}"
            }
            
        except Exception as e:
            return {"error": f"수율 분석 중 오류: {str(e)}"}
    
    @staticmethod
    def _analyze_by_lot(df: pd.DataFrame) -> Dict[str, Any]:
        """Lot별 수율 분석"""
        lot_yields = df.groupby("lot_id")["yield"].agg(['mean', 'std', 'count']).reset_index()
        lot_yields.columns = ["lot_id", "yield_mean", "yield_std", "wafer_count"]
        
        # 문제 Lot 식별
        yield_threshold = df["yield"].quantile(0.1)
        problem_lots = lot_yields[lot_yields["yield_mean"] < yield_threshold]
        
        # 최고/최저 성능 Lot
        best_lot = lot_yields.loc[lot_yields["yield_mean"].idxmax()]
        worst_lot = lot_yields.loc[lot_yields["yield_mean"].idxmin()]
        
        return {
            "total_lots": len(lot_yields),
            "problem_lots_count": len(problem_lots),
            "problem_lot_ids": problem_lots["lot_id"].tolist()[:5],
            "best_performing_lot": {
                "lot_id": best_lot["lot_id"],
                "yield": float(best_lot["yield_mean"])
            },
            "worst_performing_lot": {
                "lot_id": worst_lot["lot_id"],
                "yield": float(worst_lot["yield_mean"])
            },
            "lot_yield_variation": float(lot_yields["yield_mean"].std()),
            "avg_wafers_per_lot": float(lot_yields["wafer_count"].mean())
        }
    
    @staticmethod
    def _analyze_by_wafer(df: pd.DataFrame) -> Dict[str, Any]:
        """Wafer별 수율 분석"""
        wafer_yields = df.groupby("wafer_id")["yield"].mean()
        
        return {
            "total_wafers": len(wafer_yields),
            "wafer_yield_mean": float(wafer_yields.mean()),
            "wafer_yield_std": float(wafer_yields.std()),
            "outlier_wafers": {
                "low_yield": wafer_yields[wafer_yields < wafer_yields.quantile(0.05)].index.tolist()[:5],
                "high_yield": wafer_yields[wafer_yields > wafer_yields.quantile(0.95)].index.tolist()[:5]
            }
        }
    
    @staticmethod
    def _analyze_by_equipment(df: pd.DataFrame) -> Dict[str, Any]:
        """장비별 수율 분석"""
        equipment_performance = {}
        
        for equipment in df["equipment_id"].unique():
            equipment_data = df[df["equipment_id"] == equipment]
            
            equipment_performance[equipment] = {
                "yield_mean": float(equipment_data["yield"].mean()),
                "yield_std": float(equipment_data["yield"].std()),
                "sample_count": len(equipment_data)
            }
        
        # 장비 랭킹
        equipment_ranking = sorted(
            equipment_performance.items(),
            key=lambda x: x[1]["yield_mean"],
            reverse=True
        )
        
        return {
            "equipment_performance": equipment_performance,
            "equipment_ranking": {
                "best_equipment": equipment_ranking[0][0] if equipment_ranking else None,
                "worst_equipment": equipment_ranking[-1][0] if equipment_ranking else None,
                "performance_gap": (equipment_ranking[0][1]["yield_mean"] - 
                                  equipment_ranking[-1][1]["yield_mean"]) if len(equipment_ranking) > 1 else 0
            }
        }
    
    @staticmethod
    def _analyze_yield_distribution(yield_series: pd.Series) -> Dict[str, Any]:
        """수율 분포 분석"""
        
        # 분위수 계산
        quartiles = {
            "q1": float(yield_series.quantile(0.25)),
            "q2": float(yield_series.quantile(0.5)),
            "q3": float(yield_series.quantile(0.75))
        }
        
        # 분포 특성
        skewness = float(yield_series.skew())
        kurtosis = float(yield_series.kurtosis())
        
        # 수율 구간별 분포
        bins = [0, 0.7, 0.8, 0.9, 0.95, 1.0]
        bin_labels = ["<70%", "70-80%", "80-90%", "90-95%", "≥95%"]
        yield_binned = pd.cut(yield_series, bins=bins, labels=bin_labels, include_lowest=True)
        bin_distribution = yield_binned.value_counts().to_dict()
        
        return {
            "quartiles": quartiles,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "bin_distribution": bin_distribution,
            "distribution_type": "normal" if abs(skewness) < 0.5 else "skewed"
        }
    
    @staticmethod
    def _analyze_yield_trend(df: pd.DataFrame) -> Dict[str, Any]:
        """수율 트렌드 분석"""
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_sorted = df.sort_values("timestamp")
        
        # 일별 평균 수율
        daily_yield = df_sorted.groupby(df_sorted["timestamp"].dt.date)["yield"].mean()
        
        if len(daily_yield) < 3:
            return {"message": "트렌드 분석을 위한 충분한 데이터가 없습니다."}
        
        # 선형 트렌드 분석
        from scipy import stats
        x = np.arange(len(daily_yield))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_yield.values)
        
        # 트렌드 방향 결정
        if p_value < 0.05:
            if slope > 0:
                trend_direction = "improving"
            else:
                trend_direction = "declining"
        else:
            trend_direction = "stable"
        
        # 변동성 분석
        recent_period = daily_yield.tail(7)  # 최근 7일
        historical_period = daily_yield.head(max(7, len(daily_yield) - 7))  # 과거 데이터
        
        recent_volatility = recent_period.std() if len(recent_period) > 1 else 0
        historical_volatility = historical_period.std() if len(historical_period) > 1 else 0
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": float(abs(r_value)),
            "slope": float(slope),
            "p_value": float(p_value),
            "daily_yield_mean": float(daily_yield.mean()),
            "recent_volatility": float(recent_volatility),
            "historical_volatility": float(historical_volatility),
            "volatility_change": "increased" if recent_volatility > historical_volatility * 1.2 else "decreased" if recent_volatility < historical_volatility * 0.8 else "stable"
        }
    
    @staticmethod
    def _generate_yield_recommendations(yield_stats: Dict[str, Any]) -> List[Dict[str, str]]:
        """수율 개선 권장사항 생성"""
        
        recommendations = []
        overall_yield = yield_stats["overall_yield"]
        grade = yield_stats["performance_grade"]["grade"]
        
        if overall_yield < 0.95:
            recommendations.append({
                "priority": "high",
                "category": "yield_improvement",
                "action": "수율 개선 프로젝트 시작",
                "description": f"현재 수율 {overall_yield:.1%}를 95% 이상으로 개선",
                "timeline": "8-12주"
            })
        
        if grade in ["Poor", "Critical"]:
            recommendations.append({
                "priority": "urgent",
                "category": "immediate_action",
                "action": "긴급 품질 점검",
                "description": "공정 파라미터 및 장비 상태 즉시 점검",
                "timeline": "1-2일"
            })
        
        # 변동성이 큰 경우
        if yield_stats["yield_std"] > 0.05:
            recommendations.append({
                "priority": "medium",
                "category": "process_stability",
                "action": "공정 안정성 개선",
                "description": "수율 변동성 감소를 위한 공정 최적화",
                "timeline": "4-6주"
            })
        
        return recommendations
    
    @staticmethod
    def analyze_wafer_map_patterns(data: List[List[Union[float, int, str]]], 
                                 column_names: List[str] = None,
                                 pattern_detection: bool = True) -> Dict[str, Any]:
        """웨이퍼 맵 패턴 분석"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            required_cols = ["die_x", "die_y"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {"error": f"필수 컬럼 누락: {missing_cols}"}
            
            results = {}
            
            # 웨이퍼별 분석
            if "wafer_id" in df.columns:
                wafer_stats = {}
                
                for wafer_id in df["wafer_id"].unique():
                    wafer_data = df[df["wafer_id"] == wafer_id]
                    wafer_analysis = SemiconductorProcessTools._analyze_single_wafer_map(wafer_data, pattern_detection)
                    wafer_stats[wafer_id] = wafer_analysis
                
                results["wafer_statistics"] = wafer_stats
                
                # 전체 웨이퍼 요약
                total_wafers = len(wafer_stats)
                if "test_result" in df.columns:
                    avg_yield = df.groupby("wafer_id").apply(
                        lambda x: (x["test_result"] == "pass").sum() / len(x)
                    ).mean()
                    results["summary"] = {
                        "total_wafers": total_wafers,
                        "average_yield": float(avg_yield),
                        "yield_range": [
                            float(df.groupby("wafer_id").apply(lambda x: (x["test_result"] == "pass").sum() / len(x)).min()),
                            float(df.groupby("wafer_id").apply(lambda x: (x["test_result"] == "pass").sum() / len(x)).max())
                        ]
                    }
            else:
                # 단일 웨이퍼 분석
                single_wafer_analysis = SemiconductorProcessTools._analyze_single_wafer_map(df, pattern_detection)
                results = single_wafer_analysis
            
            # 패턴 분석 (요청시)
            if pattern_detection:
                pattern_analysis = SemiconductorProcessTools._detect_wafer_patterns(df)
                results["pattern_analysis"] = pattern_analysis
            
            return {
                "wafer_map_analysis": results,
                "analysis_metadata": {
                    "total_dies": len(df),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "pattern_detection_enabled": pattern_detection
                },
                "interpretation": f"웨이퍼 맵 분석 완료: {len(df)}개 다이 분석"
            }
            
        except Exception as e:
            return {"error": f"웨이퍼 맵 분석 중 오류: {str(e)}"}
    
    @staticmethod
    def _analyze_single_wafer_map(wafer_data: pd.DataFrame, pattern_detection: bool) -> Dict[str, Any]:
        """단일 웨이퍼 맵 분석"""
        
        analysis = {
            "total_dies": len(wafer_data),
            "die_coordinates": {
                "x_range": [int(wafer_data["die_x"].min()), int(wafer_data["die_x"].max())],
                "y_range": [int(wafer_data["die_y"].min()), int(wafer_data["die_y"].max())],
                "die_dimensions": {
                    "width": int(wafer_data["die_x"].max() - wafer_data["die_x"].min() + 1),
                    "height": int(wafer_data["die_y"].max() - wafer_data["die_y"].min() + 1)
                }
            }
        }
        
        # Pass/Fail 분석
        if "test_result" in wafer_data.columns:
            pass_dies = len(wafer_data[wafer_data["test_result"] == "pass"])
            fail_dies = len(wafer_data[wafer_data["test_result"] == "fail"])
            wafer_yield = (pass_dies / len(wafer_data)) * 100 if len(wafer_data) > 0 else 0
            
            analysis["yield_info"] = {
                "pass_dies": pass_dies,
                "fail_dies": fail_dies,
                "wafer_yield": float(wafer_yield),
                "pass_rate": float(pass_dies / len(wafer_data)) if len(wafer_data) > 0 else 0
            }
            
            # 실패 패턴 분석
            if fail_dies > 0:
                fail_pattern = SemiconductorProcessTools._analyze_fail_pattern(wafer_data)
                analysis["fail_pattern"] = fail_pattern
        
        # 중심/가장자리 분석
        if "test_result" in wafer_data.columns:
            center_edge_analysis = SemiconductorProcessTools._analyze_center_edge_pattern(wafer_data)
            analysis["center_edge_analysis"] = center_edge_analysis
        
        return analysis
    
    @staticmethod
    def _analyze_fail_pattern(wafer_data: pd.DataFrame) -> Dict[str, Any]:
        """실패 패턴 분석"""
        
        fail_data = wafer_data[wafer_data["test_result"] == "fail"]
        
        if len(fail_data) == 0:
            return {"pattern_type": "no_fails", "description": "실패 없음"}
        
        # 중심부와 가장자리 구분
        center_x = wafer_data["die_x"].mean()
        center_y = wafer_data["die_y"].mean()
        
        fail_data = fail_data.copy()
        fail_data["distance_from_center"] = np.sqrt(
            (fail_data["die_x"] - center_x)**2 + (fail_data["die_y"] - center_y)**2
        )
        
        # 전체 웨이퍼의 최대 거리
        max_distance = np.sqrt(
            (wafer_data["die_x"] - center_x)**2 + (wafer_data["die_y"] - center_y)**2
        ).max()
        
        # 가장자리 실패 비율 (외곽 30% 영역)
        edge_threshold = max_distance * 0.7
        edge_fails = len(fail_data[fail_data["distance_from_center"] > edge_threshold])
        total_fails = len(fail_data)
        edge_fail_ratio = edge_fails / total_fails if total_fails > 0 else 0
        
        # 패턴 분류
        if edge_fail_ratio > 0.7:
            pattern_type = "edge_heavy"
            description = "가장자리 집중 실패 패턴"
            recommendation = "척킹, 온도 불균일, 가스 플로우 점검"
        elif edge_fail_ratio < 0.3:
            pattern_type = "center_heavy"
            description = "중심부 집중 실패 패턴"
            recommendation = "웨이퍼 휨, 포커스 문제 점검"
        else:
            # 클러스터링 분석
            if len(fail_data) > 10:
                pattern_type = "systematic"
                description = "체계적 실패 패턴"
                recommendation = "공정 조건, 마스크 상태 점검"
            else:
                pattern_type = "random"
                description = "랜덤 실패 패턴"
                recommendation = "파티클, 재료 품질 점검"
        
        return {
            "pattern_type": pattern_type,
            "description": description,
            "edge_fail_ratio": float(edge_fail_ratio),
            "total_fails": total_fails,
            "fail_density": float(total_fails / len(wafer_data)),
            "recommendation": recommendation
        }
    
    @staticmethod
    def _analyze_center_edge_pattern(wafer_data: pd.DataFrame) -> Dict[str, Any]:
        """중심/가장자리 패턴 분석"""
        
        center_x = wafer_data["die_x"].mean()
        center_y = wafer_data["die_y"].mean()
        
        # 거리 계산
        wafer_data = wafer_data.copy()
        wafer_data["distance_from_center"] = np.sqrt(
            (wafer_data["die_x"] - center_x)**2 + (wafer_data["die_y"] - center_y)**2
        )
        
        max_distance = wafer_data["distance_from_center"].max()
        
        # 3개 영역으로 구분: 중심 (33%), 중간 (33%), 가장자리 (33%)
        center_threshold = max_distance * 0.33
        middle_threshold = max_distance * 0.67
        
        center_dies = wafer_data[wafer_data["distance_from_center"] <= center_threshold]
        middle_dies = wafer_data[
            (wafer_data["distance_from_center"] > center_threshold) & 
            (wafer_data["distance_from_center"] <= middle_threshold)
        ]
        edge_dies = wafer_data[wafer_data["distance_from_center"] > middle_threshold]
        
        # 각 영역별 수율 계산
        region_analysis = {}
        
        for region_name, region_data in [("center", center_dies), ("middle", middle_dies), ("edge", edge_dies)]:
            if len(region_data) > 0 and "test_result" in region_data.columns:
                pass_count = len(region_data[region_data["test_result"] == "pass"])
                region_yield = (pass_count / len(region_data)) * 100
                
                region_analysis[region_name] = {
                    "die_count": len(region_data),
                    "pass_count": pass_count,
                    "yield": float(region_yield)
                }
            else:
                region_analysis[region_name] = {
                    "die_count": len(region_data),
                    "pass_count": 0,
                    "yield": 0.0
                }
        
        # 수율 편차 분석
        yields = [region["yield"] for region in region_analysis.values()]
        yield_variation = np.std(yields) if len(yields) > 1 else 0
        
        # 패턴 판단
        center_yield = region_analysis["center"]["yield"]
        edge_yield = region_analysis["edge"]["yield"]
        yield_difference = center_yield - edge_yield
        
        if yield_difference > 10:
            pattern_interpretation = "중심부 우세 패턴"
        elif yield_difference < -10:
            pattern_interpretation = "가장자리 우세 패턴"
        else:
            pattern_interpretation = "균일한 패턴"
        
        return {
            "region_analysis": region_analysis,
            "yield_variation": float(yield_variation),
            "center_edge_difference": float(yield_difference),
            "pattern_interpretation": pattern_interpretation,
            "uniformity_assessment": "good" if yield_variation < 5 else "poor"
        }
    
    @staticmethod
    def _detect_wafer_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """웨이퍼 패턴 탐지"""
        
        patterns = {}
        
        if "test_result" in df.columns:
            fail_data = df[df["test_result"] == "fail"]
            
            if len(fail_data) > 5:
                # 클러스터링 분석
                try:
                    from sklearn.cluster import DBSCAN
                    
                    coordinates = fail_data[["die_x", "die_y"]].values
                    clustering = DBSCAN(eps=2, min_samples=3).fit(coordinates)
                    
                    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                    n_noise = list(clustering.labels_).count(-1)
                    
                    patterns["clustering"] = {
                        "number_of_clusters": n_clusters,
                        "noise_points": n_noise,
                        "cluster_pattern": "systematic" if n_clusters > 1 else "random"
                    }
                
                except ImportError:
                    patterns["clustering"] = {"error": "scikit-learn not available"}
                
                # 방향성 패턴 분석
                directional_patterns = SemiconductorProcessTools._analyze_directional_patterns(fail_data)
                patterns["directional"] = directional_patterns
        
        return patterns
    
    @staticmethod
    def _analyze_directional_patterns(fail_data: pd.DataFrame) -> Dict[str, Any]:
        """방향성 패턴 분석"""
        
        patterns = {}
        
        # X, Y 방향 분포 분석
        x_distribution = fail_data["die_x"].value_counts()
        y_distribution = fail_data["die_y"].value_counts()
        
        # 특정 행/열에 집중된 실패가 있는지 확인
        x_concentration = x_distribution.max() / len(fail_data) if len(fail_data) > 0 else 0
        y_concentration = y_distribution.max() / len(fail_data) if len(fail_data) > 0 else 0
        
        patterns["x_concentration"] = float(x_concentration)
        patterns["y_concentration"] = float(y_concentration)
        
        # 집중도가 높으면 라인 패턴으로 판단
        if x_concentration > 0.3:
            patterns["x_line_pattern"] = True
            patterns["dominant_x"] = int(x_distribution.idxmax())
        else:
            patterns["x_line_pattern"] = False
        
        if y_concentration > 0.3:
            patterns["y_line_pattern"] = True
            patterns["dominant_y"] = int(y_distribution.idxmax())
        else:
            patterns["y_line_pattern"] = False
        
        # 대각선 패턴 분석
        diagonal_analysis = SemiconductorProcessTools._check_diagonal_pattern(fail_data)
        patterns["diagonal"] = diagonal_analysis
        
        return patterns
    
    @staticmethod
    def _check_diagonal_pattern(fail_data: pd.DataFrame) -> Dict[str, Any]:
        """대각선 패턴 확인"""
        
        # 주 대각선 (x = y 근처)
        main_diag_fails = 0
        # 부 대각선 (x + y = constant 근처)
        anti_diag_fails = 0
        
        tolerance = 2  # ±2 범위
        
        for _, row in fail_data.iterrows():
            x, y = row["die_x"], row["die_y"]
            
            # 주 대각선 체크
            if abs(x - y) <= tolerance:
                main_diag_fails += 1
            
            # 부 대각선 체크 (임시로 x + y = 중간값 사용)
            sum_center = (fail_data["die_x"].max() + fail_data["die_y"].max()) / 2
            if abs(x + y - sum_center) <= tolerance:
                anti_diag_fails += 1
        
        total_fails = len(fail_data)
        
        return {
            "main_diagonal_fails": main_diag_fails,
            "anti_diagonal_fails": anti_diag_fails,
            "main_diagonal_ratio": main_diag_fails / total_fails if total_fails > 0 else 0,
            "anti_diagonal_ratio": anti_diag_fails / total_fails if total_fails > 0 else 0,
            "diagonal_pattern_detected": (main_diag_fails / total_fails > 0.2) or (anti_diag_fails / total_fails > 0.2)
        }
    
    @staticmethod
    def calculate_process_capability(data: List[List[float]], 
                                   column_names: List[str] = None,
                                   spec_limits: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """공정 능력 계산 (Cpk, Ppk)"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            if spec_limits is None:
                return {"error": "스펙 한계값(spec_limits)이 필요합니다."}
            
            results = {}
            
            for parameter, limits in spec_limits.items():
                if parameter not in df.columns:
                    continue
                
                param_data = df[parameter].dropna()
                
                if len(param_data) < 10:
                    results[parameter] = {"error": "공정 능력 계산을 위한 충분한 데이터가 없습니다."}
                    continue
                
                # 기본 통계
                mean = param_data.mean()
                std = param_data.std(ddof=1)  # 표본 표준편차
                
                # 스펙 한계
                usl = limits.get("usl")  # Upper Spec Limit
                lsl = limits.get("lsl")  # Lower Spec Limit
                target = limits.get("target", mean)
                
                capability_metrics = {
                    "mean": float(mean),
                    "std": float(std),
                    "target": float(target),
                    "usl": usl,
                    "lsl": lsl,
                    "sample_size": len(param_data)
                }
                
                # Cp 계산 (공정 능력 지수)
                if usl is not None and lsl is not None:
                    cp = (usl - lsl) / (6 * std) if std > 0 else float('inf')
                    capability_metrics["cp"] = float(cp) if cp != float('inf') else None
                
                # Cpk 계산 (공정 능력 지수, 중심화 고려)
                cpk_values = []
                
                if usl is not None:
                    cpu = (usl - mean) / (3 * std) if std > 0 else float('inf')
                    capability_metrics["cpu"] = float(cpu) if cpu != float('inf') else None
                    cpk_values.append(cpu)
                
                if lsl is not None:
                    cpl = (mean - lsl) / (3 * std) if std > 0 else float('inf')
                    capability_metrics["cpl"] = float(cpl) if cpl != float('inf') else None
                    cpk_values.append(cpl)
                
                if cpk_values:
                    cpk = min([val for val in cpk_values if val != float('inf')])
                    capability_metrics["cpk"] = float(cpk) if cpk != float('inf') else None
                
                # Ppk 계산 (성능 지수, 전체 분산 사용)
                # 여기서는 간단히 Cpk와 동일하게 계산 (실제로는 장기 표준편차 사용)
                capability_metrics["ppk"] = capability_metrics.get("cpk")
                
                # 수율 추정
                if usl is not None and lsl is not None:
                    from scipy import stats
                    yield_estimate = stats.norm.cdf(usl, mean, std) - stats.norm.cdf(lsl, mean, std)
                    capability_metrics["estimated_yield"] = float(yield_estimate)
                    
                    # 불량률 계산 (ppm)
                    defect_rate = 1 - yield_estimate
                    defect_ppm = defect_rate * 1e6
                    capability_metrics["defect_rate_ppm"] = float(defect_ppm)
                
                # 능력 평가
                cpk_value = capability_metrics.get("cpk", 0)
                if cpk_value is None:
                    assessment = "insufficient_data"
                elif cpk_value >= 2.0:
                    assessment = "excellent"
                elif cpk_value >= 1.67:
                    assessment = "very_good"
                elif cpk_value >= 1.33:
                    assessment = "good"
                elif cpk_value >= 1.0:
                    assessment = "acceptable"
                else:
                    assessment = "poor"
                
                capability_metrics["capability_assessment"] = assessment
                
                # 개선 권장사항
                recommendations = []
                
                if cpk_value is not None and cpk_value < 1.33:
                    if abs(mean - target) > std:
                        recommendations.append("공정 중심 조정 필요")
                    if std > (usl - lsl) / 8 if (usl and lsl) else False:
                        recommendations.append("공정 분산 감소 필요")
                
                capability_metrics["recommendations"] = recommendations
                
                results[parameter] = capability_metrics
            
            return {
                "process_capability": results,
                "analysis_summary": {
                    "parameters_analyzed": len(results),
                    "excellent_processes": len([r for r in results.values() if r.get("capability_assessment") == "excellent"]),
                    "poor_processes": len([r for r in results.values() if r.get("capability_assessment") == "poor"]),
                },
                "interpretation": f"공정 능력 분석 완료: {len(results)}개 파라미터 분석"
            }
            
        except Exception as e:
            return {"error": f"공정 능력 계산 중 오류: {str(e)}"}
    
    @staticmethod
    def perform_root_cause_analysis(data: List[List[Union[float, str]]], 
                                   column_names: List[str] = None,
                                   problem_statement: str = "",
                                   correlation_threshold: float = 0.3) -> Dict[str, Any]:
        """근본원인 분석 (Root Cause Analysis)"""
        try:
            # 데이터 변환
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data)
            
            rca_results = {
                "problem_statement": problem_statement,
                "potential_causes": [],
                "correlation_analysis": {},
                "recommendations": []
            }
            
            # 1. 상관관계 분석
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                correlation_matrix = df[numeric_cols].corr()
                
                # 강한 상관관계 식별
                strong_correlations = []
                
                for i, col1 in enumerate(correlation_matrix.columns):
                    for j, col2 in enumerate(correlation_matrix.columns):
                        if i < j:
                            corr_value = correlation_matrix.loc[col1, col2]
                            if abs(corr_value) >= correlation_threshold:
                                strong_correlations.append({
                                    "variable1": col1,
                                    "variable2": col2,
                                    "correlation": float(corr_value),
                                    "strength": "strong" if abs(corr_value) > 0.7 else "moderate",
                                    "direction": "positive" if corr_value > 0 else "negative"
                                })
                
                rca_results["correlation_analysis"] = {
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "strong_correlations": strong_correlations,
                    "parameters_analyzed": numeric_cols
                }
            
            # 2. 수율 관련 분석 (수율 데이터가 있는 경우)
            if "yield" in df.columns:
                yield_correlations = []
                
                for col in numeric_cols:
                    if col != "yield":
                        correlation = df[["yield", col]].corr().iloc[0, 1]
                        if abs(correlation) >= correlation_threshold:
                            yield_correlations.append({
                                "parameter": col,
                                "correlation_with_yield": float(correlation),
                                "impact": "positive" if correlation > 0 else "negative"
                            })
                
                # 수율에 영향을 주는 주요 파라미터
                yield_correlations.sort(key=lambda x: abs(x["correlation_with_yield"]), reverse=True)
                
                if yield_correlations:
                    rca_results["potential_causes"].append({
                        "category": "process_parameters",
                        "description": "수율에 영향을 주는 공정 파라미터",
                        "details": yield_correlations[:5],  # 상위 5개
                        "confidence": "high" if abs(yield_correlations[0]["correlation_with_yield"]) > 0.5 else "medium"
                    })
            
            # 3. 이상치 분석
            outlier_analysis = SemiconductorProcessTools._analyze_outliers_for_rca(df, numeric_cols)
            if outlier_analysis["outliers_detected"]:
                rca_results["potential_causes"].append({
                    "category": "outliers",
                    "description": "이상치 패턴 분석",
                    "details": outlier_analysis,
                    "confidence": "medium"
                })
            
            # 4. 시간적 패턴 분석 (타임스탬프가 있는 경우)
            if "timestamp" in df.columns:
                temporal_analysis = SemiconductorProcessTools._analyze_temporal_patterns_for_rca(df)
                if temporal_analysis["patterns_detected"]:
                    rca_results["potential_causes"].append({
                        "category": "temporal_patterns",
                        "description": "시간적 패턴 분석",
                        "details": temporal_analysis,
                        "confidence": "medium"
                    })
            
            # 5. 장비별 분석 (장비 정보가 있는 경우)
            if "equipment_id" in df.columns:
                equipment_analysis = SemiconductorProcessTools._analyze_equipment_for_rca(df)
                if equipment_analysis["variations_detected"]:
                    rca_results["potential_causes"].append({
                        "category": "equipment_variation",
                        "description": "장비간 성능 편차",
                        "details": equipment_analysis,
                        "confidence": "high"
                    })
            
            # 6. 권장사항 생성
            rca_results["recommendations"] = SemiconductorProcessTools._generate_rca_recommendations(
                rca_results["potential_causes"]
            )
            
            return {
                "root_cause_analysis": rca_results,
                "analysis_metadata": {
                    "potential_causes_found": len(rca_results["potential_causes"]),
                    "correlation_threshold": correlation_threshold,
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "interpretation": f"근본원인 분석 완료: {len(rca_results['potential_causes'])}개 잠재 원인 식별"
            }
            
        except Exception as e:
            return {"error": f"근본원인 분석 중 오류: {str(e)}"}
    
    @staticmethod
    def _analyze_outliers_for_rca(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """RCA용 이상치 분석"""
        
        outlier_info = {}
        total_outliers = 0
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) > 10:
                # IQR 방법으로 이상치 탐지
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_info[col] = {
                        "outlier_count": outlier_count,
                        "outlier_percentage": (outlier_count / len(col_data)) * 100,
                        "outlier_values": outliers.tolist()[:5]  # 최대 5개만
                    }
                    total_outliers += outlier_count
        
        return {
            "outliers_detected": total_outliers > 0,
            "total_outliers": total_outliers,
            "outlier_details": outlier_info
        }
    
    @staticmethod
    def _analyze_temporal_patterns_for_rca(df: pd.DataFrame) -> Dict[str, Any]:
        """RCA용 시간적 패턴 분석"""
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        patterns = {}
        patterns_detected = False
        
        # 수율 트렌드 분석
        if "yield" in df.columns:
            df_sorted = df.sort_values("timestamp")
            daily_yield = df_sorted.groupby(df_sorted["timestamp"].dt.date)["yield"].mean()
            
            if len(daily_yield) >= 5:
                # 최근 기간과 이전 기간 비교
                recent_period = daily_yield.tail(max(3, len(daily_yield) // 3))
                earlier_period = daily_yield.head(max(3, len(daily_yield) // 3))
                
                recent_avg = recent_period.mean()
                earlier_avg = earlier_period.mean()
                
                yield_change = recent_avg - earlier_avg
                
                if abs(yield_change) > 0.02:  # 2% 이상 변화
                    patterns["yield_trend"] = {
                        "change_direction": "decline" if yield_change < 0 else "improvement",
                        "change_magnitude": float(abs(yield_change)),
                        "recent_avg": float(recent_avg),
                        "earlier_avg": float(earlier_avg)
                    }
                    patterns_detected = True
        
        # 주기적 패턴 분석
        if len(df) > 20:
            # 요일별 패턴
            df["weekday"] = df["timestamp"].dt.dayofweek
            
            if "yield" in df.columns:
                weekday_yield = df.groupby("weekday")["yield"].mean()
                weekday_variation = weekday_yield.std()
                
                if weekday_variation > 0.01:  # 1% 이상 편차
                    patterns["weekday_pattern"] = {
                        "variation_detected": True,
                        "weekday_yields": weekday_yield.to_dict(),
                        "variation_magnitude": float(weekday_variation)
                    }
                    patterns_detected = True
        
        return {
            "patterns_detected": patterns_detected,
            "temporal_patterns": patterns
        }
    
    @staticmethod
    def _analyze_equipment_for_rca(df: pd.DataFrame) -> Dict[str, Any]:
        """RCA용 장비 분석"""
        
        equipment_analysis = {}
        variations_detected = False
        
        if "yield" in df.columns:
            equipment_yields = df.groupby("equipment_id")["yield"].agg(['mean', 'std', 'count'])
            
            # 장비간 편차 분석
            yield_variation = equipment_yields["mean"].std()
            
            if yield_variation > 0.02:  # 2% 이상 편차
                variations_detected = True
                
                # 최고/최저 성능 장비
                best_equipment = equipment_yields["mean"].idxmax()
                worst_equipment = equipment_yields["mean"].idxmin()
                
                equipment_analysis = {
                    "yield_variation": float(yield_variation),
                    "best_equipment": {
                        "equipment_id": best_equipment,
                        "yield": float(equipment_yields.loc[best_equipment, "mean"])
                    },
                    "worst_equipment": {
                        "equipment_id": worst_equipment,
                        "yield": float(equipment_yields.loc[worst_equipment, "mean"])
                    },
                    "performance_gap": float(
                        equipment_yields.loc[best_equipment, "mean"] - 
                        equipment_yields.loc[worst_equipment, "mean"]
                    )
                }
        
        return {
            "variations_detected": variations_detected,
            "equipment_details": equipment_analysis
        }
    
    @staticmethod
    def _generate_rca_recommendations(potential_causes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """RCA 기반 권장사항 생성"""
        
        recommendations = []
        
        for cause in potential_causes:
            category = cause["category"]
            confidence = cause["confidence"]
            
            if category == "process_parameters":
                recommendations.append({
                    "priority": "high" if confidence == "high" else "medium",
                    "action": "공정 파라미터 최적화",
                    "description": "수율에 큰 영향을 주는 파라미터들의 조건 재검토",
                    "timeline": "2-4주"
                })
            
            elif category == "equipment_variation":
                recommendations.append({
                    "priority": "high",
                    "action": "장비 성능 균일화",
                    "description": "저성능 장비의 PM 및 캘리브레이션 실시",
                    "timeline": "1-2주"
                })
            
            elif category == "temporal_patterns":
                recommendations.append({
                    "priority": "medium",
                    "action": "시간적 변화 요인 분석",
                    "description": "공정 드리프트 및 환경 변화 요인 조사",
                    "timeline": "2-3주"
                })
            
            elif category == "outliers":
                recommendations.append({
                    "priority": "medium",
                    "action": "이상치 원인 조사",
                    "description": "측정 시스템 점검 및 특이 조건 식별",
                    "timeline": "1-2주"
                })
        
        return recommendations


# MCP 도구 등록
@mcp.tool
def analyze_yield_performance(data: List[List[Union[float, str]]], 
                            column_names: List[str] = None,
                            analysis_scope: str = "lot",
                            target_yield: float = 0.95) -> Dict[str, Any]:
    """
    반도체 수율 성능을 분석합니다.
    
    Args:
        data: 분석할 데이터 (2차원 리스트)
        column_names: 컬럼 이름들
        analysis_scope: 분석 범위 (lot, wafer, equipment)
        target_yield: 목표 수율 (기본값: 0.95)
    
    Returns:
        수율 성능 분석 결과
    """
    return SemiconductorProcessTools.analyze_yield_performance(data, column_names, analysis_scope, target_yield)


@mcp.tool
def analyze_wafer_map_patterns(data: List[List[Union[float, int, str]]], 
                             column_names: List[str] = None,
                             pattern_detection: bool = True) -> Dict[str, Any]:
    """
    웨이퍼 맵 패턴을 분석합니다.
    
    Args:
        data: 웨이퍼 맵 데이터 (die_x, die_y, test_result 포함)
        column_names: 컬럼 이름들
        pattern_detection: 패턴 탐지 수행 여부
    
    Returns:
        웨이퍼 맵 패턴 분석 결과
    """
    return SemiconductorProcessTools.analyze_wafer_map_patterns(data, column_names, pattern_detection)


@mcp.tool
def calculate_process_capability(data: List[List[float]], 
                               column_names: List[str] = None,
                               spec_limits: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
    """
    공정 능력을 계산합니다 (Cp, Cpk, Ppk).
    
    Args:
        data: 공정 데이터
        column_names: 컬럼 이름들
        spec_limits: 스펙 한계값 (예: {"parameter": {"usl": 10, "lsl": 0, "target": 5}})
    
    Returns:
        공정 능력 계산 결과
    """
    return SemiconductorProcessTools.calculate_process_capability(data, column_names, spec_limits)


@mcp.tool
def perform_root_cause_analysis(data: List[List[Union[float, str]]], 
                               column_names: List[str] = None,
                               problem_statement: str = "",
                               correlation_threshold: float = 0.3) -> Dict[str, Any]:
    """
    근본원인 분석을 수행합니다.
    
    Args:
        data: 분석할 데이터
        column_names: 컬럼 이름들
        problem_statement: 문제 정의
        correlation_threshold: 상관관계 임계값 (기본값: 0.3)
    
    Returns:
        근본원인 분석 결과
    """
    return SemiconductorProcessTools.perform_root_cause_analysis(data, column_names, problem_statement, correlation_threshold)


@mcp.tool
def generate_spc_control_limits(data: List[float], 
                              parameter_name: str = "parameter",
                              control_type: str = "xbar_r") -> Dict[str, Any]:
    """
    SPC 관리한계를 생성합니다.
    
    Args:
        data: 공정 데이터 (1차원 리스트)
        parameter_name: 파라미터 이름
        control_type: 관리도 유형 (xbar_r, individual, cusum)
    
    Returns:
        SPC 관리한계 결과
    """
    try:
        data_series = pd.Series(data)
        
        if len(data_series) < 20:
            return {"error": "SPC 관리한계 계산을 위해서는 최소 20개 이상의 데이터가 필요합니다."}
        
        # 기본 통계
        mean = data_series.mean()
        std = data_series.std()
        
        # 관리한계 계산
        if control_type == "individual":
            # 개별값 관리도
            mr = data_series.diff().abs().mean()  # Moving Range 평균
            ucl = mean + 2.66 * mr
            lcl = mean - 2.66 * mr
        else:
            # 일반적인 3σ 관리한계
            ucl = mean + 3 * std
            lcl = mean - 3 * std
        
        # 경고한계 (2σ)
        uwl = mean + 2 * std
        lwl = mean - 2 * std
        
        # 관리상태 점검
        out_of_control = []
        warning_points = []
        
        for i, value in enumerate(data_series):
            if value > ucl or value < lcl:
                out_of_control.append(i)
            elif value > uwl or value < lwl:
                warning_points.append(i)
        
        # 공정 능력 (단순 계산)
        if std > 0:
            # 6σ가 스펙 범위라고 가정 (임시)
            cp_estimate = 1.0  # 기본값
        else:
            cp_estimate = float('inf')
        
        return {
            "spc_control_limits": {
                "parameter_name": parameter_name,
                "control_type": control_type,
                "center_line": float(mean),
                "ucl": float(ucl),
                "lcl": float(lcl),
                "uwl": float(uwl),
                "lwl": float(lwl)
            },
            "control_violations": {
                "out_of_control_points": out_of_control,
                "warning_points": warning_points,
                "violation_rate": len(out_of_control) / len(data_series)
            },
            "process_assessment": {
                "in_control": len(out_of_control) == 0,
                "capability_estimate": float(cp_estimate) if cp_estimate != float('inf') else None,
                "data_points": len(data_series)
            },
            "interpretation": f"SPC 관리한계 계산 완료: {len(out_of_control)}개 관리한계 이탈점 발견"
        }
        
    except Exception as e:
        return {"error": f"SPC 관리한계 계산 중 오류: {str(e)}"}


@mcp.tool
def comprehensive_semiconductor_analysis(data: List[List[Union[float, str]]], 
                                       column_names: List[str] = None,
                                       analysis_types: List[str] = None,
                                       target_yield: float = 0.95) -> Dict[str, Any]:
    """
    종합적인 반도체 공정 분석을 수행합니다.
    
    Args:
        data: 분석할 데이터
        column_names: 컬럼 이름들
        analysis_types: 수행할 분석 유형 리스트 (yield, wafer_map, capability, rca)
        target_yield: 목표 수율
    
    Returns:
        종합 반도체 분석 결과
    """
    try:
        if analysis_types is None:
            analysis_types = ["yield", "capability", "rca"]
        
        # 데이터 변환
        if column_names:
            df = pd.DataFrame(data, columns=column_names)
        else:
            df = pd.DataFrame(data)
        
        comprehensive_results = {"comprehensive_analysis": {}}
        
        # 수율 분석
        if "yield" in analysis_types and "yield" in df.columns:
            yield_result = SemiconductorProcessTools.analyze_yield_performance(
                data, column_names, "lot", target_yield
            )
            comprehensive_results["comprehensive_analysis"]["yield_analysis"] = yield_result
        
        # 웨이퍼 맵 분석
        if "wafer_map" in analysis_types and all(col in df.columns for col in ["die_x", "die_y"]):
            wafer_map_result = SemiconductorProcessTools.analyze_wafer_map_patterns(
                data, column_names, True
            )
            comprehensive_results["comprehensive_analysis"]["wafer_map_analysis"] = wafer_map_result
        
        # 공정 능력 분석
        if "capability" in analysis_types:
            # 기본 스펙 한계 생성 (실제로는 사용자 제공 필요)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                spec_limits = {}
                for col in numeric_cols[:3]:  # 최대 3개 파라미터
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        mean = col_data.mean()
                        std = col_data.std()
                        spec_limits[col] = {
                            "usl": mean + 3 * std,
                            "lsl": mean - 3 * std,
                            "target": mean
                        }
                
                if spec_limits:
                    capability_result = SemiconductorProcessTools.calculate_process_capability(
                        data, column_names, spec_limits
                    )
                    comprehensive_results["comprehensive_analysis"]["capability_analysis"] = capability_result
        
        # 근본원인 분석
        if "rca" in analysis_types and len(df.select_dtypes(include=[np.number]).columns) >= 2:
            rca_result = SemiconductorProcessTools.perform_root_cause_analysis(
                data, column_names, "반도체 공정 품질 이슈", 0.3
            )
            comprehensive_results["comprehensive_analysis"]["root_cause_analysis"] = rca_result
        
        # 요약 정보
        comprehensive_results["analysis_summary"] = {
            "total_samples": len(df),
            "parameters_analyzed": len(df.columns),
            "analyses_performed": analysis_types,
            "target_yield": target_yield,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return comprehensive_results
        
    except Exception as e:
        return {"error": f"종합 반도체 분석 중 오류: {str(e)}"}


if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Semiconductor Process Tools MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)