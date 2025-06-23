# -*- coding: utf-8 -*-
"""
MCP Tool ▶ A2A-compliant Data Science Suite
This module has been refactored to use the a2a-sdk, exposing its functionalities
as standard A2A skills. The core logic from LDSuite remains, but it is now
structured around the A2AServer for standardized agent-to-agent communication.
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

# A2A-SDK for serving skills
from a2a.server import A2AServer
from a2a.model import Response, Content

# Optional packages
try: import xgboost as xgb; XGB = True
except ImportError: XGB = False
try: import lightgbm as lgb; LGB = True
except ImportError: LGB = False
try: import catboost as cb; CAT = True
except ImportError: CAT = False
try: import statsmodels.api as sm; from statsmodels.tsa.statespace.sarimax import SARIMAX; SM = True
except ImportError: SM = False
try: import shap; SHAP = True
except ImportError: SHAP = False
try: import optuna; OPT = True
except ImportError: OPT = False


# ──────────────────────────────────────────────────────────────
# Server and Path Configuration
# ──────────────────────────────────────────────────────────────
SERVER_PORT = int(os.getenv("SERVER_PORT", "8007"))
ROOT        = Path(os.getenv("SANDBOX_DIR", "./sandbox")).resolve()
DATA_DIR    = ROOT / "datasets";  DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR   = ROOT / "models"  ;  MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR    = ROOT / "plots"   ;  PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR     = ROOT / "logs"    ;  LOG_DIR.mkdir(exist_ok=True)

# Initialize A2A Server
server = A2AServer(port=SERVER_PORT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────
# Utilities & Tracker (unchanged)
# ──────────────────────────────────────────────────────────────
class _Tracker:
    def __init__(self): self.ops: Dict[str, Dict[str, Any]] = {}
    def log(self, op: str, kind: str, inp: Dict[str, Any], out: Dict[str, Any], files: List[str]):
        rec = dict(id=op, kind=kind, time=datetime.utcnow().isoformat(), inp=inp, out=out, files=files)
        self.ops[op] = rec
        with open(LOG_DIR / f"op_{op}.json", "w", encoding="utf8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
tracker = _Tracker()

def _id(tag: str) -> str: return f"{tag}_{uuid.uuid4().hex[:8]}"
def _save_df(df: pd.DataFrame, name: str) -> Path:
    p = DATA_DIR / f"{name}.csv"; df.to_csv(p, index=False); return p
def _save_fig(op: str, name: str) -> str:
    path = PLOT_DIR / f"{op}_{name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close(); return str(path)

# ──────────────────────────────────────────────────────────────
# Core Skills (Refactored from LDSuite class)
# ──────────────────────────────────────────────────────────────

@server.skill()
def upload(path: str) -> Response:
    """Uploads a file to the server's dataset directory."""
    if not Path(path).exists():
        return Response(status="error", message=f"File not found: {path}")
    dst = DATA_DIR / Path(path).name; shutil.copy2(path, dst)
    op = _id("upload")
    out = {"saved_as": dst.name, "size_mb": round(dst.stat().st_size / 1e6, 3)}
    tracker.log(op, "upload", {"src": path}, out, [str(dst)])
    return Response(status="success", message="File uploaded.", contents=[Content(data=out)])

@server.skill()
def load(file: str, sample: int = 0) -> Response:
    """Loads a CSV file into a DataFrame and saves it with a new dataset ID."""
    try:
        p = DATA_DIR / file
        df = pd.read_csv(p)
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=42)
        ds = _id("ds"); _save_df(df, ds)
        out = {"dataset_id": ds, "rows": len(df), "cols": len(df.columns)}
        tracker.log(ds, "load", {"file": file, "sample": sample}, out, [str(DATA_DIR / f"{ds}.csv")])
        return Response(status="success", message="Dataset loaded.", contents=[Content(data=out)])
    except Exception as e:
        return Response(status="error", message=str(e))

@server.skill()
def eda(ds_id: str, max_hist: int = 3) -> Response:
    """Performs a quick exploratory data analysis on a dataset."""
    try:
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
        op = _id("eda"); plots: List[str] = []
        num = df.select_dtypes(include=[np.number]).columns[:max_hist]
        for c in num:
            df[c].hist(bins=30, alpha=.7); plt.title(f"{c} dist")
            plots.append(_save_fig(op, c))
        out = {"plots": plots, "describe": df.describe().to_dict()}
        tracker.log(op, "eda", {"ds": ds_id}, out, plots)
        return Response(status="success", message="EDA complete.", contents=[Content(data=out)])
    except Exception as e:
        return Response(status="error", message=str(e))

@server.skill()
def viz(ds_id: str, kind: str, x: str, y: Optional[str] = None) -> Response:
    """Generates a visualization from a dataset."""
    try:
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv"); op = _id("viz")
        if kind == "scatter" and y:
            plt.scatter(df[x], df[y], alpha=.6); plt.xlabel(x); plt.ylabel(y)
        else:
            df[x].hist(bins=30, alpha=.7); plt.xlabel(x); plt.ylabel("freq")
        p = _save_fig(op, f"{kind}_{x}{'_' + y if y else ''}")
        out = {"plot": p}
        tracker.log(op, "viz", {"ds": ds_id, "kind": kind, "x": x, "y": y}, out, [p])
        return Response(status="success", message="Visualization created.", contents=[Content(data=out)])
    except Exception as e:
        return Response(status="error", message=str(e))

@server.skill()
def train(ds_id: str, target: str, boost: bool = False) -> Response:
    """Trains a machine learning model."""
    try:
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
        if target not in df.columns:
            raise KeyError("target column missing")
        X, y = df.drop(columns=[target]), df[target]
        X = pd.get_dummies(X, drop_first=True)
        feature_cols = list(X.columns)
        Xtr, Xte, ytr, yte = model_selection.train_test_split(
            X, y, test_size=.2, random_state=42,
            stratify=y if y.nunique() < 20 else None)
        cls = y.nunique() < 20
        metric = metrics.accuracy_score if cls else metrics.r2_score
        models: Dict[str, Any] = {
            "rf": ensemble.RandomForestClassifier(200, random_state=42) if cls else ensemble.RandomForestRegressor(200, random_state=42),
            "gb": ensemble.GradientBoostingClassifier(random_state=42) if cls else ensemble.GradientBoostingRegressor(random_state=42),
            "linear": linear_model.LogisticRegression(max_iter=1000) if cls else linear_model.LinearRegression(),
        }
        if boost and XGB: models["xgb"] = (xgb.XGBClassifier(random_state=42, eval_metric="logloss") if cls else xgb.XGBRegressor(random_state=42))
        if boost and LGB: models["lgb"] = (lgb.LGBMClassifier(random_state=42) if cls else lgb.LGBMRegressor(random_state=42))
        if boost and CAT: models["cat"] = (cb.CatBoostClassifier(verbose=False, random_state=42) if cls else cb.CatBoostRegressor(verbose=False, random_state=42))

        best_s: float = 0.0 if cls else -np.inf
        best_m: Any = None
        scores: Dict[str, float] = {}
        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            s = metric(yte, mdl.predict(Xte))
            scores[name] = s
            if (cls and s > best_s) or (not cls and s > best_s):
                best_s, best_m = s, mdl

        model_bundle = {"model": best_m, "columns": feature_cols}
        mid = _id("model"); joblib.dump(model_bundle, MODEL_DIR / f"{mid}.pkl")
        op = _id("train")
        out = {"model_id": mid, "best_score": best_s, "scores": scores}
        tracker.log(op, "train", {"ds": ds_id, "target": target, "boost": boost}, out, [str(MODEL_DIR / f"{mid}.pkl")])
        return Response(status="success", message="Model trained.", contents=[Content(data=out)])
    except Exception as e:
        return Response(status="error", message=str(e))

@server.skill()
def tune(ds_id: str, target: str, n_trials: int = 30) -> Response:
    """Tunes hyperparameters for a model using Optuna."""
    if not OPT: return Response(status="error", message="optuna not installed")
    try:
        df = pd.read_csv(DATA_DIR / f"{ds_id}.csv")
        X, y = pd.get_dummies(df.drop(columns=[target]), drop_first=True), df[target]
        cls = y.nunique() < 20
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            lr = trial.suggest_float("learning_rate", 0.01, 0.3)
            mdl = (ensemble.GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, random_state=42) if cls
                   else ensemble.GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, random_state=42))
            score = model_selection.cross_val_score(mdl, X, y, cv=3, scoring="accuracy" if cls else "r2").mean()
            return score
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        mdl_best = (ensemble.GradientBoostingClassifier(**best_params, random_state=42) if cls
                    else ensemble.GradientBoostingRegressor(**best_params, random_state=42))
        mdl_best.fit(X, y)
        mid = _id("tuned_model"); joblib.dump(mdl_best, MODEL_DIR / f"{mid}.pkl")
        op = _id("tune")
        out = {"model_id": mid, "best_params": best_params, "best_value": study.best_value}
        tracker.log(op, "tune", {"ds": ds_id, "target": target, "n_trials": n_trials}, out, [str(MODEL_DIR / f"{mid}.pkl")])
        return Response(status="success", message="Tuning complete.", contents=[Content(data=out)])
    except Exception as e:
        return Response(status="error", message=str(e))

def _run_step(step: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """Helper to run a single step of an orchestration pipeline."""
    action = step["action"]
    params = step.get("params", {})
    
    # Resolve dependencies from context
    for p_name, p_value in params.items():
        if isinstance(p_value, str) and p_value.startswith("{{") and p_value.endswith("}}"):
            key = p_value[2:-2].strip()  # e.g., "load_step.dataset_id"
            step_name, attr = key.split('.')
            if step_name in context:
                params[p_name] = context[step_name].get(attr)

    skill_func = server.skills.get(action)
    if not skill_func:
        raise ValueError(f"Skill '{action}' not found.")
    
    # Execute the skill and get the data from the first content object
    response = skill_func.callback(**params)
    if response.status == "error":
        raise RuntimeError(f"Step '{action}' failed: {response.message}")
    return response.contents[0].data if response.contents else {}

@server.skill()
def run_orchestration(flow: Union[str, List[Dict[str, Any]]], continue_on_error: bool = True) -> Response:
    """Runs a pipeline of skills defined in a JSON/YAML flow."""
    try:
        if isinstance(flow, str):
            if flow.strip().startswith("[") or flow.strip().startswith("{"):
                flow_data = json.loads(flow)
            else:
                flow_data = yaml.safe_load(flow)
        else:
            flow_data = flow

        steps = flow_data if isinstance(flow_data, list) else flow_data.get("steps", [])
        context: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []

        for step in steps:
            step_id = step["id"]
            try:
                result_data = _run_step(step, context)
                context[step_id] = result_data
                results.append({"step": step_id, "status": "success", "output": result_data})
            except Exception as e:
                logger.error(f"Error in step '{step_id}': {e}")
                results.append({"step": step_id, "status": "error", "message": str(e)})
                if not continue_on_error:
                    return Response(status="error", message=f"Pipeline failed at step '{step_id}'.", contents=[Content(data=results)])
        
        return Response(status="success", message="Orchestration finished.", contents=[Content(data=results)])
    except Exception as e:
        return Response(status="error", message=f"Failed to run orchestration: {e}")

if __name__ == "__main__":
    print(f"🚀 Starting A2A Data Science Server at port {SERVER_PORT}...")
    server.run()
