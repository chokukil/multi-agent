# -*- coding: utf-8 -*-
"""
MCP Tool ▶ Defect Pattern Analysis

웨이퍼 맵·결함·Lot 데이터를 입체적으로 분석해
패턴 식별, 클러스터링, 밀도·지역 통계, Pareto, Trend 등을 제공합니다.
깨졌던 한글과 문법 오류를 전면 복구했습니다.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import uvicorn
from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------
# 서버 설정
# ------------------------------------------------------------
SERVER_PORT = int(os.getenv("SERVER_PORT", "8011"))

mcp = FastMCP("Defect Pattern Analysis")

# ------------------------------------------------------------
# 핵심 분석 클래스
# ------------------------------------------------------------
class DefectPatternAnalyzer:
    """결함 패턴 분석 유틸리티"""

    # -----------------------------------------------------
    # 1. 웨이퍼 맵 패턴 분석
    # -----------------------------------------------------
    @staticmethod
    def analyze_wafer_map_patterns(wafer_map: List[List[int]], die_size: Tuple[float, float] = (10.0, 10.0)) -> Dict[str, Any]:
        if not wafer_map or not all(isinstance(r, list) for r in wafer_map):
            return {"error": "wafer_map 형식이 잘못되었습니다."}
        arr = np.asarray(wafer_map, dtype=int)
        rows, cols = arr.shape
        total = rows * cols
        uniques, counts = np.unique(arr, return_counts=True)
        dist = dict(zip(uniques.tolist(), counts.tolist()))
        pass_cnt = dist.get(1, 0)
        fail_cnt = total - pass_cnt
        yield_pct = pass_cnt / total * 100

        spatial = DefectPatternAnalyzer._detect_spatial_patterns(arr)
        clusters = DefectPatternAnalyzer._find_defect_clusters(arr)
        regions = DefectPatternAnalyzer._analyze_wafer_regions(arr)
        density = DefectPatternAnalyzer._calculate_defect_density(arr, die_size)

        return {
            "wafer_statistics": {
                "size": f"{rows}x{cols}",
                "total_dies": total,
                "pass_count": pass_cnt,
                "fail_count": fail_cnt,
                "yield_percent": round(yield_pct, 2),
                "die_distribution": dist,
            },
            "spatial_patterns": spatial,
            "cluster_analysis": clusters,
            "region_analysis": regions,
            "defect_density": density,
        }

    # -----------------------------------------------------
    # 1‑1. 공간 패턴 감지
    # -----------------------------------------------------
    @staticmethod
    def _detect_spatial_patterns(arr: np.ndarray) -> Dict[str, Any]:
        rows, cols = arr.shape
        patterns: List[Dict[str, Any]] = []

        # Edge vs Center 비교
        edge_w = max(1, min(rows, cols) // 10)
        edge_mask = np.zeros_like(arr, dtype=bool)
        edge_mask[:edge_w, :] = edge_mask[-edge_w:, :] = True
        edge_mask[:, :edge_w] = edge_mask[:, -edge_w:] = True
        edge_fail = np.sum((arr == 0) & edge_mask) / edge_mask.sum() if edge_mask.sum() else 0
        center_fail = np.sum((arr == 0) & ~edge_mask) / (~edge_mask).sum() if (~edge_mask).sum() else 0
        if edge_fail > center_fail * 1.5:
            patterns.append({"type": "Edge Effect", "edge_fail_rate_pct": round(edge_fail*100,2), "center_fail_rate_pct": round(center_fail*100,2)})

        # Center core
        cy, cx = rows // 2, cols // 2
        r_core = min(rows, cols) // 4
        yy, xx = np.ogrid[:rows, :cols]
        mask_core = (yy - cy)**2 + (xx - cx)**2 <= r_core**2
        core_fail = np.sum((arr == 0) & mask_core) / mask_core.sum() if mask_core.sum() else 0
        outer_fail = np.sum((arr == 0) & ~mask_core) / (~mask_core).sum() if (~mask_core).sum() else 0
        if core_fail > outer_fail * 1.5:
            patterns.append({"type": "Center Defect", "center_fail_rate_pct": round(core_fail*100,2)})

        # Ring pattern
        patterns.extend(DefectPatternAnalyzer._detect_ring_patterns(arr))
        # Linear pattern
        patterns.extend(DefectPatternAnalyzer._detect_linear_patterns(arr))

        return {
            "detected_patterns": patterns,
            "pattern_count": len(patterns),
            "dominant_pattern": patterns[0]["type"] if patterns else "Random"
        }

    # Ring 패턴
    @staticmethod
    def _detect_ring_patterns(arr: np.ndarray) -> List[Dict[str, Any]]:
        rows, cols = arr.shape
        cy, cx = rows // 2, cols // 2
        max_r = min(rows, cols) // 2
        ring_w = max(2, max_r // 10)
        overall_fail = np.mean(arr == 0)
        patterns = []
        for r in range(ring_w, max_r, ring_w):
            yy, xx = np.ogrid[:rows, :cols]
            ring_mask = ((yy - cy)**2 + (xx - cx)**2 < r**2) & ((yy - cy)**2 + (xx - cx)**2 >= (r-ring_w)**2)
            if not ring_mask.any():
                continue
            ring_fail = np.sum((arr == 0) & ring_mask) / ring_mask.sum()
            if ring_fail > overall_fail * 1.8:
                patterns.append({"type": "Ring Pattern", "radius": r, "ring_fail_rate_pct": round(ring_fail*100,2)})
        return patterns

    # Linear (가로/세로) 패턴
    @staticmethod
    def _detect_linear_patterns(arr: np.ndarray) -> List[Dict[str, Any]]:
        patterns = []
        overall_fail = np.mean(arr == 0)
        # Horizontal & Vertical
        for axis, name in [(0, "Horizontal"), (1, "Vertical")]:
            for idx in range(arr.shape[axis]):
                line = arr[idx, :] if axis == 0 else arr[:, idx]
                fail_rate = np.mean(line == 0)
                if fail_rate > max(0.3, overall_fail * 2):
                    patterns.append({"type": f"{name} Line", "index": idx, "fail_rate_pct": round(fail_rate*100,2)})
        return patterns

    # -----------------------------------------------------
    # 1‑2. 결함 클러스터 탐지
    # -----------------------------------------------------
    @staticmethod
    def _find_defect_clusters(arr: np.ndarray) -> Dict[str, Any]:
        visited = np.zeros_like(arr, dtype=bool)
        clusters: List[List[Tuple[int, int]]] = []
        rows, cols = arr.shape

        def flood(r: int, c: int):
            stack, cl = [(r, c)], []
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= rows or x < 0 or x >= cols or visited[y, x] or arr[y, x] != 0:
                    continue
                visited[y, x] = True
                cl.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy or dx:
                            stack.append((y+dy, x+dx))
            return cl

        for r in range(rows):
            for c in range(cols):
                if arr[r, c] == 0 and not visited[r, c]:
                    cl = flood(r, c)
                    if len(cl) > 1:
                        clusters.append(cl)

        if not clusters:
            return {"cluster_count": 0}

        sizes = [len(cl) for cl in clusters]
        largest = max(clusters, key=len)
        total_def = int(np.sum(arr == 0))
        total_clust = sum(sizes)
        analyses = []
        for rank, cl in enumerate(sorted(clusters, key=len, reverse=True)[:5], start=1):
            ys, xs = zip(*cl)
            dens = len(cl) / ((max(ys)-min(ys)+1)*(max(xs)-min(xs)+1))
            analyses.append({"rank": rank, "size": len(cl), "center": (round(np.mean(ys),1), round(np.mean(xs),1)), "density": round(dens,3)})

        return {
            "cluster_count": len(clusters),
            "largest_cluster_size": len(largest),
            "cluster_coverage_pct": round(total_clust / total_def * 100, 2) if total_def else 0,
            "average_cluster_size": round(float(np.mean(sizes)), 2),
            "cluster_analysis": analyses,
        }

    # -----------------------------------------------------
    # 1‑3. 3×3 지역 분석
    # -----------------------------------------------------
    @staticmethod
    def _analyze_wafer_regions(arr: np.ndarray) -> Dict[str, Any]:
        rows, cols = arr.shape
        r_split = [rows//3, rows//3, rows - 2*(rows//3)]
        c_split = [cols//3, cols//3, cols - 2*(cols//3)]
        names = [["Top-Left","Top-Center","Top-Right"],["Mid-Left","Center","Mid-Right"],["Bottom-Left","Bottom-Center","Bottom-Right"]]
        regions: Dict[str, Any] = {}
        r0 = 0
        for i in range(3):
            c0 = 0
            for j in range(3):
                r1, c1 = r0 + r_split[i], c0 + c_split[j]
                sub = arr[r0:r1, c0:c1]
                fail = np.mean(sub == 0) * 100 if sub.size else 0
                regions[names[i][j]] = {"fail_rate_pct": round(fail,2), "coords": f"({r0},{c0})-({r1-1},{c1-1})"}
                c0 = c1
            r0 = r1
        fails = [v["fail_rate_pct"] for v in regions.values()]
        best = min(regions, key=lambda k: regions[k]["fail_rate_pct"])
        worst = max(regions, key=lambda k: regions[k]["fail_rate_pct"])
        return {
            "regions": regions,
            "best_region": best,
            "worst_region": worst,
            "uniformity_cv_pct": round(float(np.std(fails)/np.mean(fails)*100),2) if np.mean(fails) else 0
        }

    # -----------------------------------------------------
    # 1‑4. 결함 밀도
    # -----------------------------------------------------
    @staticmethod
    def _calculate_defect_density(arr: np.ndarray, die_size: Tuple[float, float]) -> Dict[str, Any]:
        rows, cols = arr.shape
        dw, dh = die_size
        wafer_area_mm2 = rows * cols * dw * dh
        def_cnt = int(np.sum(arr == 0))
        dens_cm2 = def_cnt / (wafer_area_mm2 / 100) if wafer_area_mm2 else 0
        exp_yield = math.exp(-dens_cm2 * (dw*dh/100)) * 100
        return {"defect_density_per_cm2": round(dens_cm2,4), "expected_yield_pct": round(exp_yield,2), "total_defects": def_cnt}

    # -----------------------------------------------------
    # 2. 결함 시그니처 분류
    # -----------------------------------------------------
    @staticmethod
    def classify_defect_signatures(defects: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not defects:
            return {"error": "defect 데이터가 없습니다."}
        types = Counter(d.get("type", "Unknown") for d in defects)
        sizes = [d["size"] for d in defects if isinstance(d.get("size"), (int, float))]
        severities = Counter(d.get("severity", "Unknown") for d in defects)
        locs = [d["location"] for d in defects if isinstance(d.get("location"), (list, tuple)) and len(d["location"])==2]
        size_stats = {"mean": round(np.mean(sizes),2) if sizes else 0, "std": round(np.std(sizes),2) if sizes else 0}
        spatial = {}
        if locs:
            xs, ys = zip(*locs)
            spatial = {
                "center_of_mass": [round(np.mean(xs),2), round(np.mean(ys),2)],
                "spread_x": round(float(np.std(xs)),2),
                "spread_y": round(float(np.std(ys)),2),
            }
        total = len(defects)
        pareto, cum = [], 0.0
        for t, cnt in types.most_common():
            pct = cnt/total*100
            cum += pct
            pareto.append({"type": t, "count": cnt, "percent": round(pct,2), "cum_percent": round(cum,2)})
        vital = [p for p in pareto if p["cum_percent"] <= 80]
        return {
            "total_defects": total,
            "classification": {"by_type": types, "by_severity": severities},
            "size_stats": size_stats,
            "spatial_stats": spatial,
            "pareto": pareto,
            "vital_few_types": [v["type"] for v in vital],
        }

    # -----------------------------------------------------
    # 3. Lot 레벨 분석
    # -----------------------------------------------------
    @staticmethod
    def analyze_lot_level_patterns(lots: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not lots:
            return {"error": "lot 데이터가 없습니다."}
        ylds = [l.get("yield", 0.0) for l in lots]
        defs = [l.get("defect_count", 0) for l in lots]
        stats = {"mean_yield": round(float(np.mean(ylds)),2), "std_yield": round(float(np.std(ylds)),2)}
        # Trend
        trend = {"direction": "Insufficient"}
        if len(ylds) > 2:
            x = np.arange(len(ylds))
            slope, _ = np.polyfit(x, ylds, 1)
            trend = {
                "slope": round(float(slope),4),
                "direction": "Improving" if slope>0.1 else "Declining" if slope<-0.1 else "Stable"
            }
        # Outliers
        z = (ylds - np.mean(ylds)) / (np.std(ylds) or 1)
        outs = [{"lot_id": l.get("lot_id", i), "yield": ylds[i]} for i,l in enumerate(lots) if abs(z[i])>2]
        return {"lot_stats": stats, "trend": trend, "outliers": outs}

# ------------------------------------------------------------
# MCP 래퍼
# ------------------------------------------------------------
@mcp.tool("analyze_wafer_map_patterns")
def analyze_wafer_map_patterns(wafer_map: str, die_width: float = 10.0, die_height: float = 10.0) -> Dict[str, Any]:
    try:
        return DefectPatternAnalyzer.analyze_wafer_map_patterns(json.loads(wafer_map), (die_width, die_height))
    except json.JSONDecodeError:
        return {"error": "wafer_map JSON 형식 오류"}

@mcp.tool("classify_defect_signatures")
def classify_defect_signatures(defect_data: str) -> Dict[str, Any]:
    try:
        return DefectPatternAnalyzer.classify_defect_signatures(json.loads(defect_data))
    except json.JSONDecodeError:
        return {"error": "defect_data JSON 형식 오류"}

@mcp.tool("analyze_lot_level_patterns")
def analyze_lot_level_patterns(lot_data: str) -> Dict[str, Any]:
    try:
        return DefectPatternAnalyzer.analyze_lot_level_patterns(json.loads(lot_data))
    except json.JSONDecodeError:
        return {"error": "lot_data JSON 형식 오류"}

# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting Defect Pattern Analysis MCP server on port {SERVER_PORT}…")
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=SERVER_PORT, log_level="info")
