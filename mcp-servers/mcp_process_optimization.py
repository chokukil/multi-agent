# -*- coding: utf-8 -*-
"""
MCP Tool ▶ Process Optimization
DOE(실험계획) 설계·분석과 반응표면 최적화, 공정 파라미터 개선 로드맵을 지원합니다.
한글이 깨졌던 주석·문자열을 복구하고, 문법 오류·따옴표 누락을 전면 수정했습니다.
"""

from __future__ import annotations

import json
import os
from itertools import combinations, product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import uvicorn
from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------
# 서버 환경 변수
# ------------------------------------------------------------
SERVER_PORT = int(os.getenv("SERVER_PORT", "8012"))

mcp = FastMCP("Process Optimization")

# ------------------------------------------------------------
# 핵심 클래스
# ------------------------------------------------------------
class ProcessOptimizer:
    """DOE 설계·분석 & 반응표면 최적화 유틸리티"""

    # --------------------------------------------------------
    # 1. DOE 설계
    # --------------------------------------------------------
    @staticmethod
    def design_factorial_experiment(
        factors: Dict[str, Dict[str, float]],
        design_type: str = "full_factorial",
    ) -> Dict[str, Any]:
        """DOE 실험 행렬 생성 (Full / Fractional / Central Composite)"""
        if not factors:
            return {"error": "factors 입력이 비어 있습니다."}

        factor_levels: Dict[str, List[float]] = {}
        for name, info in factors.items():
            low, high = info.get("low"), info.get("high")
            if low is None or high is None:
                return {"error": f"factor '{name}'에 low/high 값이 필요합니다."}
            center = info.get("center", (low + high) / 2)
            if design_type == "full_factorial":
                factor_levels[name] = [low, high]
            elif design_type == "central_composite":
                alpha = info.get("alpha", 1.414)
                factor_levels[name] = [low, center, high, center - alpha * (high - center) / 2, center + alpha * (high - center) / 2]
            else:  # fractional_factorial or default
                factor_levels[name] = [low, center, high]

        # 실험 조합 생성
        experiments: List[Dict[str, Any]] = []
        names = list(factor_levels.keys())

        if design_type == "full_factorial":
            for levels in product(*factor_levels.values()):
                experiments.append({k: v for k, v in zip(names, levels)})

        elif design_type == "fractional_factorial":
            base_combos = list(product([0, 1], repeat=len(names) - 1))
            for combo in base_combos:
                exp = {names[i]: factor_levels[names[i]][lvl] for i, lvl in enumerate(combo)}
                interaction = sum(combo) % 2
                exp[names[-1]] = factor_levels[names[-1]][interaction]
                experiments.append(exp)

        else:  # central_composite
            factorial_pts = list(product(*[levels[:2] for levels in factor_levels.values()]))
            for levels in factorial_pts:
                experiments.append({k: v for k, v in zip(names, levels)})
            center = {k: v[1] for k, v in factor_levels.items()}
            experiments.extend([center.copy() for _ in range(3)])  # center points
            for i, name in enumerate(names):
                for idx in [3, 4]:
                    axial = center.copy()
                    axial[name] = factor_levels[name][idx]
                    experiments.append(axial)

        np.random.shuffle(experiments)
        for idx, exp in enumerate(experiments, start=1):
            exp["run_order"] = idx

        return {
            "design_type": design_type,
            "total_runs": len(experiments),
            "factors": names,
            "factor_levels": factor_levels,
            "experiment_matrix": experiments,
            "design_properties": {
                "balanced": True,
                "orthogonal": design_type in {"full_factorial", "central_composite"},
                "resolution": "V" if design_type == "full_factorial" else "III" if design_type == "fractional_factorial" else "N/A",
            },
        }

    # --------------------------------------------------------
    # 2. DOE 결과 분석
    # --------------------------------------------------------
    @staticmethod
    def analyze_doe_results(experiment_data: List[Dict[str, Any]], response: str) -> Dict[str, Any]:
        if not experiment_data:
            return {"error": "experiment_data 가 비어 있습니다."}
        df = pd.DataFrame(experiment_data)
        if response not in df.columns:
            return {"error": f"response 컬럼 '{response}' 이(가) 없습니다."}
        factor_cols = [c for c in df.columns if c not in {response, "run_order"}]
        if not factor_cols:
            return {"error": "factor 컬럼을 찾을 수 없습니다."}

        y = df[response].to_numpy()
        y_mean, y_std = float(y.mean()), float(y.std())

        # 주효과
        main_effects: Dict[str, Any] = {}
        for col in factor_cols:
            levels = np.unique(df[col])
            if len(levels) < 2:
                continue
            low, high = levels[0], levels[-1]
            effect = float(df.loc[df[col] == high, response].mean() - df.loc[df[col] == low, response].mean())
            main_effects[col] = {
                "effect": round(effect, 4),
                "significance": "High" if abs(effect) > y_std else "Low",
            }

        # 2차 상호작용
        interaction_effects: Dict[str, Any] = {}
        for a, b in combinations(factor_cols, 2):
            combo_mean = df.groupby([a, b])[response].mean().reset_index()
            levels_a = sorted(df[a].unique())[:2]
            levels_b = sorted(df[b].unique())[:2]
            try:
                hh = combo_mean[(combo_mean[a] == levels_a[1]) & (combo_mean[b] == levels_b[1])][response].iloc[0]
                ll = combo_mean[(combo_mean[a] == levels_a[0]) & (combo_mean[b] == levels_b[0])][response].iloc[0]
                hl = combo_mean[(combo_mean[a] == levels_a[1]) & (combo_mean[b] == levels_b[0])][response].iloc[0]
                lh = combo_mean[(combo_mean[a] == levels_a[0]) & (combo_mean[b] == levels_b[1])][response].iloc[0]
                interaction = ((hh + ll) - (hl + lh)) / 2
                interaction_effects[f"{a}*{b}"] = {
                    "interaction_effect": round(float(interaction), 4),
                    "significance": "High" if abs(interaction) > y_std * 0.5 else "Low",
                }
            except IndexError:
                continue

        ss_total = float(np.sum((y - y_mean) ** 2))
        ss_explained = sum((v["effect"] ** 2) * len(y) / 4 for v in main_effects.values())
        r2 = ss_explained / ss_total if ss_total else 0.0

        recs: List[str] = []
        if main_effects:
            key_factor = max(main_effects, key=lambda k: abs(main_effects[k]["effect"]))
            recs.append(f"주효과가 큰 요인: '{key_factor}'")
        sig_inters = [k for k, v in interaction_effects.items() if v["significance"] == "High"]
        if sig_inters:
            recs.append("의미 있는 상호작용: " + ", ".join(sig_inters))

        return {
            "experiment_summary": {
                "total_runs": len(df),
                "factors": factor_cols,
                "response": response,
            },
            "main_effects": main_effects,
            "interaction_effects": interaction_effects,
            "model_performance": {
                "r_squared": round(r2, 4),
                "model_adequacy": "Good" if r2 > 0.7 else "Fair" if r2 > 0.5 else "Poor",
            },
            "recommendations": recs,
        }

    # --------------------------------------------------------
    # 3. 반응표면 최적화 (Grid Search)
    # --------------------------------------------------------
    @staticmethod
    def optimize_response_surface(
        factor_ranges: Dict[str, Tuple[float, float]],
        response_model: Dict[str, float],
        goal: str = "maximize",
    ) -> Dict[str, Any]:
        if not factor_ranges or not response_model:
            return {"error": "factor_ranges 및 response_model 이 필요합니다."}

        factors = list(factor_ranges.keys())
        grid_size = 20
        grids = {f: np.linspace(r[0], r[1], grid_size) for f, r in factor_ranges.items()}

        def calc_resp(cond: Dict[str, float]) -> float:
            r = response_model.get("intercept", 0.0)
            for f in factors:
                if f in response_model:
                    r += response_model[f] * cond[f]
                quad_key = f"{f}^2"
                if quad_key in response_model:
                    r += response_model[quad_key] * cond[f] ** 2
            for a, b in combinations(factors, 2):
                inter_key = f"{a}*{b}"
                if inter_key in response_model:
                    r += response_model[inter_key] * cond[a] * cond[b]
            return r

        best_resp = -np.inf if goal == "maximize" else np.inf
        best_cond: Dict[str, float] = {}
        results: List[Dict[str, Any]] = []

        for idx in range(grid_size ** len(factors)):
            combo = {}
            tmp = idx
            for f in factors:
                combo[f] = grids[f][tmp % grid_size]
                tmp //= grid_size
            r = calc_resp(combo)
            results.append({"conditions": combo, "response": r})
            if (goal == "maximize" and r > best_resp) or (goal == "minimize" and r < best_resp):
                best_resp, best_cond = r, combo.copy()

        sorted_res = sorted(results, key=lambda x: x["response"], reverse=(goal == "maximize"))[:5]

        # 민감도 분석
        sens: Dict[str, Any] = {}
        for f in factors:
            rng = factor_ranges[f][1] - factor_ranges[f][0]
            delta = rng * 0.1
            low_cond = best_cond.copy(); low_cond[f] = max(factor_ranges[f][0], best_cond[f] - delta)
            high_cond = best_cond.copy(); high_cond[f] = min(factor_ranges[f][1], best_cond[f] + delta)
            sens_val = (calc_resp(high_cond) - calc_resp(low_cond)) / (2 * delta)
            sens[f] = {"sensitivity": round(sens_val, 4)}

        return {
            "goal": goal,
            "optimal_conditions": {k: round(v, 4) for k, v in best_cond.items()},
            "optimal_response": round(best_resp, 4),
            "alternatives": [
                {"conditions": {k: round(v, 4) for k, v in res["conditions"].items()}, "response": round(res["response"], 4)}
                for res in sorted_res
            ],
            "sensitivity_analysis": sens,
        }


# ------------------------------------------------------------
# MCP 래퍼
# ------------------------------------------------------------
@mcp.tool("design_factorial_experiment")
def design_factorial_experiment(factors: str, design_type: str = "full_factorial") -> Dict[str, Any]:
    try:
        return ProcessOptimizer.design_factorial_experiment(json.loads(factors), design_type)
    except json.JSONDecodeError:
        return {"error": "factors JSON 형식 오류"}


@mcp.tool("analyze_doe_results")
def analyze_doe_results(experiment_data: str, response_column: str) -> Dict[str, Any]:
    try:
        return ProcessOptimizer.analyze_doe_results(json.loads(experiment_data), response_column)
    except json.JSONDecodeError:
        return {"error": "experiment_data JSON 형식 오류"}


@mcp.tool("optimize_response_surface")
def optimize_response_surface(factor_ranges: str, response_model: str, goal: str = "maximize") -> Dict[str, Any]:
    try:
        return ProcessOptimizer.optimize_response_surface(json.loads(factor_ranges), json.loads(response_model), goal)
    except json.JSONDecodeError:
        return {"error": "factor_ranges 또는 response_model JSON 형식 오류"}


# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting Process Optimization MCP server on port {SERVER_PORT}…")
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=SERVER_PORT, log_level="info")
