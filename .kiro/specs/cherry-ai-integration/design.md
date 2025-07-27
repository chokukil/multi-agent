# Design Document — Agent Map & Endpoints
_(format matched to original design.md)_

## Overview
CherryAI operates a set of A2A-compliant agents with clear **ports, endpoints, and responsibilities**. This section refines the agent map to enable deterministic validation and observability without changing the architecture.

### Current System Snapshot (2025-07-26)
- ✅ A2A Orchestrator and 11 agent servers: baseline complete
- ✅ Chat-style Streamlit UI (port 8501) with SSE
- ⚠️ Langfuse v2: EMP_NO binding, trace aggregation improvements
- ⚠️ E2E (Playwright MCP): implement agent health and regression suites

## Architecture
- **Gateway/UI**: Streamlit (default **8501**)
- **Orchestrator**: A2A SDK 0.2.9 with parallel/sequential plans
- **Agents**: 11 services; discovery via `agents.json`
- **Tracing**: Langfuse v2 per session/agent
- **Error Handling**: progressive retry and circuit breaker

## Agent Map (Ports & Endpoints)
> **Authoritative source**: `config/agents.json`. The following reflect documented defaults.

| Agent              | Port  | Endpoint                         | Primary Capability                         |
|--------------------|:-----:|----------------------------------|--------------------------------------------|
| Data Cleaning      | 8306 | http://localhost:8306 | Missing/Outliers/Duplicates/Type checks    |
| Data Loader        | 8307 | http://localhost:8307 | File/DB loading, chunking                  |
| Data Visualization | 8308 | http://localhost:8308 | Plotly figures                             |
| Data Wrangling     | 8309 | http://localhost:8309 | Transform/Reshape                          |
| Feature Engineering| 8310 | http://localhost:8310 | Feature creation/selection/dim. reduction  |
| SQL Database       | 8311 | http://localhost:8311 | SQL querying & analysis                    |
| EDA Tools          | 8312 | http://localhost:8312 | Descriptive stats, correlations            |
| H2O ML             | 8313 | http://localhost:8313 | AutoML & model eval                        |
| MLflow Tools       | 8314 | http://localhost:8314 | Experiment tracking & registry             |
| Pandas Analyst     | 8315* | http://localhost:8315 | Inspect/Select/Manipulate/Aggregate/Merge  |
| Report Generator   | 8316 | http://localhost:8316 | Executive/Detailed/Quality/Stats/Visual    |

\* Some configurations use **8210** for Pandas Analyst. Confirm via `agents.json` and healthcheck.

## Healthcheck Contract
Each agent exposes:
- `GET /.well-known/agent.json` → metadata (name, version, methods)
- `POST /` JSON-RPC 2.0 → methods and params

### Example
```bash
curl -s http://localhost:8316/.well-known/agent.json | jq
curl -s http://localhost:8316 -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":"health","method":"health_check","params":{}
}' | jq
```

## Per-Agent Method Hints (Non-exhaustive)
- **Data Cleaning**: `detect_missing_values`, `handle_outliers`, `drop_duplicates`, `validate_types`
- **Data Loader**: `load_csv_files`, `load_excel_files`, `load_json_files`, `connect_db`
- **Visualization**: `generate_plotly_chart`, `export_figure`
- **Wrangling**: `transform_data`, `pivot`, `unpivot`, `merge`
- **Feature Eng.**: `create_features`, `select_features`, `reduce_dimensions`
- **SQL**: `execute_sql`, `analyze_query`
- **EDA**: `compute_descriptive_stats`, `correlation_matrix`
- **H2O ML**: `train_automl`, `evaluate_model`
- **MLflow**: `log_metrics`, `register_model`, `compare_runs`
- **Pandas Analyst**: `inspect_data`, `select_data`, `manipulate_data`, `aggregate_data`, `merge_data`
- **Report Generator**: `generate_executive_summary`, `generate_detailed_analysis`, `generate_statistical_report`

## Validation Flow (Design Perspective)
1) **Discovery**: Load `agents.json` and check all ports are reachable.
2) **Health**: Verify `agent.json` and `health_check` per agent.
3) **Smoke**: Run 1 method per agent with a public sample (`data/sample.csv`).
4) **Trace**: Ensure Langfuse v2 has session+agent spans.
5) **E2E**: Upload → Analyze → Report with Playwright MCP.

## Non-Functional Considerations
- **Performance**: qwen3-4b-fast ~45s average response target
- **Reliability**: retry budget, circuit break threshold tuning
- **Security**: input validation, method allow-list, sandboxed IO

## Change Log
- 2025-07-26: Consolidated agent ports and endpoints; added healthcheck contract and validation flow.