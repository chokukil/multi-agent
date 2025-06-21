# -*- coding: utf-8 -*-
"""
MCP Tool ▶ Lightweight Data Science Suite (Plus)
딥러닝 의존성(TensorFlow·PyTorch) 제거 후 **경량 ML·EDA·시각화·해석** 기능을 한 곳에 모은 MCP 툴킷입니다.

v1.1 (⚡ Orchestration Extension)
────────────────────────────────────────────────────────
- 🧩 **Orchestrator**: JSON/YAML 플로우 정의만으로 `upload→load→eda→train→explain`
  같은 워크플로를 순차 실행.
- 🔄 `run_pipeline` MCP tool: 파이프라인 입력 → 각 스텝 실행, 결과 집계.
- 🛡️ `continue_on_error` 옵션으로 스텝 오류 발생 시에도 다음 스텝 진행.
- 🔗 기존 **LDSuite** API(Upload / Load / EDA / Viz / Train / Tune / Forecast / Explain) 그대로 유지.
"""

from __future__ import annotations

import json, logging, os, uuid, shutil, traceback, warnings, joblib, yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble, linear_model, metrics, model_selection
import uvicorn

from mcp.server.fastmcp import FastMCP

# ──────────────────────────────────────────────────────────────
# 서버/경로 설정
# ──────────────────────────────────────────────────────────────
SERVER_PORT = int(os.getenv("SERVER_PORT", "8007"))
ROOT        = Path(os.getenv("SANDBOX_DIR", "./sandbox")).resolve()
DATA_DIR    = ROOT / "datasets";  DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR   = ROOT / "models"  ;  MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR    = ROOT / "plots"   ;  PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR     = ROOT / "logs"    ;  LOG_DIR.mkdir(exist_ok=True)

mcp = FastMCP("LightweightDataSciencePlus")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────
# 선택적 외부 패키지 존재 여부 확인 – Lazy Import
# ──────────────────────────────────────────────────────────────
XGB = LGB = CAT = False
try:
    import xgboost as xgb; XGB = True
except ImportError:
    pass
try:
    import lightgbm as lgb; LGB = True
except ImportError:
    pass
try:
    import catboost as cb; CAT = True
except ImportError:
    pass

# Advanced analytics (optional)
try:
    import statsmodels.api as sm; from statsmodels.tsa.statespace.sarimax import SARIMAX; SM = True
except ImportError:
    SM = False
try:
    import shap; SHAP = True
except ImportError:
    SHAP = False
try:
    import optuna; OPT = True
except ImportError:
    OPT = False

# ──────────────────────────────────────────────────────────────
# 유틸리티 & Tracker
# ──────────────────────────────────────────────────────────────
class _Tracker:
    def __init__(self):
        self.ops: Dict[str, Dict[str, Any]] = {}
    def log(self, op: str, kind: str, inp: Dict[str, Any], out: Dict[str, Any], files: List[str]):
        rec = dict(id=op, kind=kind, time=datetime.utcnow().isoformat(),
                   inp=inp, out=out, files=files)
        self.ops[op] = rec
        with open(LOG_DIR / f"op_{op}.json", "w", encoding="utf8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
tracker = _Tracker()

def _id(tag: str) -> str:          # 간단한 UUID helper
    return f"{tag}_{uuid.uuid4().hex[:8]}"

def _save_df(df: pd.DataFrame, name: str) -> Path:
    p = DATA_DIR / f"{name}.csv"; df.to_csv(p, index=False); return p

def _save_fig(op: str, name: str) -> str:
    path = PLOT_DIR / f"{op}_{name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close(); return str(path)

# ──────────────────────────────────────────────────────────────
# 핵심 클래스: LDSuite
# ──────────────────────────────────────────────────────────────
class LDSuite:
    """Upload / EDA / Viz / Train / Tune / Forecast / Explain 기능 모음"""

    # ── Upload / Load ─────────────────────────────────────
    @staticmethod
    def upload(path: str) -> Dict[str, Any]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        dst = DATA_DIR / Path(path).name; shutil.copy2(path, dst)
        op = _id("upload")
        out = {"saved_as": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 3)}
        tracker.log(op, "upload", {"src": path}, out, [str(dst)])
        return out

    @staticmethod
    def load(file: str, sample: int = 0) -> Dict[str, Any]:
        p = DATA_DIR / file
        df = pd.read_csv(p)
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=42)
        ds = _id("ds"); _save_df(df, ds)
        out = {"dataset_id": ds, "rows": len(df), "cols": len(df.columns)}
        tracker.log(ds, "load", {"file": file, "sample": sample},
                    out, [str(DATA_DIR / f"{ds}.csv")])
        return out

    # ── Quick EDA ─────────────────────────────────────────
    @staticmethod
    def eda(ds_id: str, max_hist: int = 3) -> Dict[str, Any]:
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
        op = _id("eda"); plots: List[str] = []
        num = df.select_dtypes(include=[np.number]).columns[:max_hist]
        for c in num:
            df[c].hist(bins=30, alpha=.7); plt.title(f"{c} dist")
            plots.append(_save_fig(op, c))
        out = {"plots": plots, "describe": df.describe().to_dict()}
        tracker.log(op, "eda", {"ds": ds_id}, out, plots)
        return out

    # ── Visualization ────────────────────────────────────
    @staticmethod
    def viz(ds_id: str, kind: str, x: str,
            y: Optional[str] = None) -> Dict[str, Any]:
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv"); op = _id("viz")
        if kind == "scatter" and y:
            plt.scatter(df[x], df[y], alpha=.6); plt.xlabel(x); plt.ylabel(y)
        else:
            df[x].hist(bins=30, alpha=.7); plt.xlabel(x); plt.ylabel("freq")
        p = _save_fig(op, f"{kind}_{x}{'_' + y if y else ''}")
        out = {"plot": p}
        tracker.log(op, "viz",
                    {"ds": ds_id, "kind": kind, "x": x, "y": y}, out, [p])
        return out

    # ── ML Train (simple) ─────────────────────────────────
    @staticmethod
    def train(ds_id: str, target: str, boost: bool = False) -> Dict[str, Any]:
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
        if target not in df.columns:
            raise KeyError("target column missing")
        X, y = df.drop(columns=[target]), df[target]
        X = pd.get_dummies(X, drop_first=True)
        feature_cols = list(X.columns)                       # 컬럼 순서 보존
        Xtr, Xte, ytr, yte = model_selection.train_test_split(
            X, y, test_size=.2, random_state=42,
            stratify=y if y.nunique() < 20 else None)
        cls = y.nunique() < 20
        metric = metrics.accuracy_score if cls else metrics.r2_score

        models: Dict[str, Any] = {
            "rf": ensemble.RandomForestClassifier(200, random_state=42)
            if cls else ensemble.RandomForestRegressor(200, random_state=42),
            "gb": ensemble.GradientBoostingClassifier(random_state=42)
            if cls else ensemble.GradientBoostingRegressor(random_state=42),
            "linear": linear_model.LogisticRegression(max_iter=1000)
            if cls else linear_model.LinearRegression(),
        }
        if boost and XGB:
            models["xgb"] = (
                xgb.XGBClassifier(random_state=42, eval_metric="logloss")
                if cls else xgb.XGBRegressor(random_state=42)
            )
        if boost and LGB:
            models["lgb"] = (
                lgb.LGBMClassifier(random_state=42)
                if cls else lgb.LGBMRegressor(random_state=42)
            )
        if boost and CAT:
            models["cat"] = (
                cb.CatBoostClassifier(verbose=False, random_state=42)
                if cls else cb.CatBoostRegressor(verbose=False, random_state=42)
            )

        best_s: float = 0.0 if cls else -np.inf               # 가독성 개선
        best_m: Any = None
        scores: Dict[str, float] = {}

        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            s = metric(yte, mdl.predict(Xte))
            scores[name] = s
            if (cls and s > best_s) or (not cls and s > best_s):
                best_s, best_m = s, mdl

        # 🌟 모델과 컬럼 순서 함께 저장
        model_bundle = {"model": best_m, "columns": feature_cols}
        mid = _id("model"); joblib.dump(model_bundle, MODEL_DIR / f"{mid}.pkl")
        op = _id("train")
        out = {"model_id": mid, "best_score": best_s, "scores": scores}
        tracker.log(op, "train", {"ds": ds_id, "target": target, "boost": boost},
                    out, [str(MODEL_DIR / f"{mid}.pkl")])
        return out

    # ── Hyper-parameter Tuning (Optuna) ──────────────────
    @staticmethod
    def tune(ds_id: str, target: str, n_trials: int = 30) -> Dict[str, Any]:
        if not OPT:
            raise ModuleNotFoundError("optuna not installed")
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
        X, y = pd.get_dummies(df.drop(columns=[target]), drop_first=True), df[target]
        cls = y.nunique() < 20

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            max_depth   = trial.suggest_int("max_depth", 3, 10)
            lr          = trial.suggest_float("learning_rate", 0.01, 0.3)
            if cls:
                mdl = ensemble.GradientBoostingClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=lr, random_state=42)
                score = model_selection.cross_val_score(
                    mdl, X, y, cv=3, scoring="accuracy").mean()
            else:
                mdl = ensemble.GradientBoostingRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=lr, random_state=42)
                score = model_selection.cross_val_score(
                    mdl, X, y, cv=3, scoring="r2").mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params

        mdl_best = (ensemble.GradientBoostingClassifier(**best_params, random_state=42)
                    if cls else
                    ensemble.GradientBoostingRegressor(**best_params, random_state=42))
        mdl_best.fit(X, y)

        bundle = {"model": mdl_best, "columns": list(X.columns),
                  "tuned_params": best_params}
        mid = _id("tuned"); joblib.dump(bundle, MODEL_DIR / f"{mid}.pkl")
        op = _id("tune")
        out = {"model_id": mid, "best_params": best_params,
               "best_value": study.best_value}
        tracker.log(op, "tune",
                    {"ds": ds_id, "target": target, "trials": n_trials},
                    out, [str(MODEL_DIR / f"{mid}.pkl")])
        return out

    # ── Forecasting (Statsmodels) ─────────────────────────
    @staticmethod
    def forecast(ds_id: str, date_col: str, target: str,
                 steps: int = 30, order: Tuple[int, int, int] = (1, 1, 1)
                ) -> Dict[str, Any]:
        if not SM:
            raise ModuleNotFoundError("statsmodels not installed")
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
        df[date_col] = pd.to_datetime(df[date_col]); df.sort_values(date_col, inplace=True)
        s = df.set_index(date_col)[target]

        model = SARIMAX(s, order=order, enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=steps).predicted_mean
        preds_dict = pred.iloc[-steps:].to_dict()   # 빈 dict 방지

        op = _id("fc")
        out = {"steps": steps, "predictions": preds_dict}
        tracker.log(op, "forecast",
                    {"ds": ds_id, "target": target, "order": order, "steps": steps},
                    out, [])
        return out

    # ── Explain Model (SHAP) ─────────────────────────────
    @staticmethod
    def explain(model_id: str, ds_id: Optional[str] = None,
                max_display: int = 10) -> Dict[str, Any]:
        if not SHAP:
            raise ModuleNotFoundError("shap not installed")
        bundle: Dict = joblib.load(MODEL_DIR / f"{model_id}.pkl")
        model, cols = bundle["model"], bundle["columns"]

        if ds_id is None:
            X_sample = pd.DataFrame([np.zeros(len(cols))], columns=cols)
        else:
            df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
            X = pd.get_dummies(df.drop(columns=[c for c in df.columns if c not in cols]),
                               drop_first=True)
            X = X.reindex(columns=cols, fill_value=0)    # 컬럼 순서·수 맞추기
            X_sample = X.sample(min(200, len(X)), random_state=42)

        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)

        op = _id("shap")
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plot_path = _save_fig(op, "shap_bar")

        out = {"plot": plot_path, "n_samples": len(X_sample)}
        tracker.log(op, "explain", {"model": model_id, "ds": ds_id},
                    out, [plot_path])
        return out

# ──────────────────────────────────────────────────────────────
# 경량 Orchestrator
# ──────────────────────────────────────────────────────────────
def _run_step(step: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """단일 스텝 실행 (함수명 + 매개변수)"""
    func = step.get("tool")
    params = step.get("params", {})
    params_resolved = {
        k: context.get(v[1:], v) if isinstance(v, str) and v.startswith("$") else v
        for k, v in params.items()
    }
    tools_map = {
        "upload": LDSuite.upload,
        "load": LDSuite.load,
        "eda": LDSuite.eda,
        "viz": LDSuite.viz,
        "train": LDSuite.train,
        "tune": LDSuite.tune,
        "forecast": LDSuite.forecast,
        "explain": LDSuite.explain,
    }
    if func not in tools_map:
        raise ValueError(f"Unknown tool: {func}")
    return tools_map[func](**params_resolved)

def run_orchestration(flow: Union[str, List[Dict[str, Any]]],
                      continue_on_error: bool = True) -> Dict[str, Any]:
    """JSON/YAML or list[dict] 파이프라인 실행"""
    if isinstance(flow, str):
        with open(flow, "r", encoding="utf8") as f:
            flow_def = yaml.safe_load(f)
    else:
        flow_def = flow
    results, ctx = [], {}
    for idx, step in enumerate(flow_def, 1):
        try:
            res = _run_step(step, ctx)
            results.append({"step": idx, "tool": step["tool"], "result": res})
            if step.get("save_as"):
                ctx[step["save_as"]] = res
        except Exception as e:
            err = traceback.format_exc()
            results.append({"step": idx, "tool": step["tool"],
                            "error": str(e), "trace": err})
            if not continue_on_error:
                break
    return {"results": results, "context": ctx}

# ──────────────────────────────────────────────────────────────
# MCP Wrapper 함수
# ──────────────────────────────────────────────────────────────
@mcp.tool()         #  upload_file(path)
def upload_file(path: str):
    return LDSuite.upload(path)

@mcp.tool()         #  load_dataset(file, sample=0)
def load_dataset(file: str, sample: int = 0):
    return LDSuite.load(file, sample)

@mcp.tool()         #  quick_eda(dataset_id)
def quick_eda(dataset_id: str):
    return LDSuite.eda(dataset_id)

@mcp.tool()         #  visualize(dataset_id, kind, x, y=None)
def visualize(dataset_id: str, kind: str, x: str, y: Optional[str] = None):
    return LDSuite.viz(dataset_id, kind, x, y)

@mcp.tool()         #  train_model(dataset_id, target, boost=False)
def train_model(dataset_id: str, target: str, boost: bool = False):
    return LDSuite.train(dataset_id, target, boost)

@mcp.tool()         #  tune_model(dataset_id, target, n_trials=30)
def tune_model(dataset_id: str, target: str, n_trials: int = 30):
    return LDSuite.tune(dataset_id, target, n_trials)

@mcp.tool()         #  forecast_series(dataset_id, date_col, target, steps)
def forecast_series(dataset_id: str, date_col: str, target: str, steps: int = 30):
    return LDSuite.forecast(dataset_id, date_col, target, steps)

@mcp.tool()         #  explain_model(model_id, dataset_id=None)
def explain_model(model_id: str, dataset_id: Optional[str] = None):
    return LDSuite.explain(model_id, dataset_id)

# ── Orchestrator 라우팅 ─────────────────────────────────────
@mcp.tool()         # run_pipeline(flow_json_or_yaml, continue_on_error=True)
def run_pipeline(flow: Union[str, List[Dict[str, Any]]],
                 continue_on_error: bool = True):
    """
    flow: list[dict] or YAML/JSON filepath
      예) [
            {"tool":"upload","params":{"path":"./data.csv"},"save_as":"u"},
            {"tool":"load","params":{"file":"$u[saved_as]"},"save_as":"d"},
            {"tool":"train","params":{"ds_id":"$d[dataset_id]","target":"label"}}
          ]
    $표현식은 직전 save_as 결과를 참조
    """
    return run_orchestration(flow, continue_on_error)

# ──────────────────────────────────────────────────────────────
# 서버 실행 스크립트
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Lightweight Data Science Suite (Plus) server…")
    logger.info(f"Sandbox root : {ROOT}")
    logger.info(f"Server port  : {SERVER_PORT}")
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=SERVER_PORT)
