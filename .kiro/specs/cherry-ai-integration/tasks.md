# Implementation Plan — Agent Validation Addendum
_(format matched to original tasks.md)_

## Overview
This addendum augments the existing implementation plan with **agent-by-agent validation tasks**. It preserves the original tone and sectioning style while adding **ports, endpoints, and module entrypoints** so you can verify each capability end-to-end.

## Key Validated Patterns to Leverage
- **A2AAgentDiscoverySystem**: Proven discovery across ports **8306–8316**
- **A2AWorkflowOrchestrator**: Sequential/parallel execution plans
- **A2AErrorHandler**: Progressive retry & circuit breaker
- **SessionManager + Langfuse v2**: Per-agent traceability
- **SSE Streaming (0.001s)**: Real-time multi-agent progress

## Enhanced UI/UX Hooks
- Chat input → Agent plan → Streaming results
- Inline errors with graceful fallback per agent
- E2E Test anchors (data-testid) for Playwright MCP

---

## Agent Validation Tasks (Checklists)

> **Endpoints** assume `http://localhost:<port>`; confirm with `config/agents.json` if customized.  
> For Pandas Analyst, some documents referenced **8210** as a dedicated port; prefer **8315** unless your `agents.json` specifies otherwise.

### 1) Data Cleaning (port 8306)
- **Endpoint**: `http://localhost:8306`
- **Capabilities**: missing/outliers/duplicates/type-check/standardization/rule-validation
- **Healthcheck**
  ```bash
  curl -s http://localhost:8306/.well-known/agent.json | jq
  ```
- **Validate**: detect missing
  ```bash
  curl -s http://localhost:8306 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"dc-1","method":"detect_missing_values","params":{"data_path":"data/sample.csv"}}' | jq
  ```
- **Pass Criteria**: returns `missing_summary` or equivalent counts.

### 2) Data Loader (port 8307)
- **Endpoint**: `http://localhost:8307`
- **Note**: Some notes mention historical use of `http://localhost:8001`. Verify active port via `agents.json`.
- **Validate**: CSV load
  ```bash
  curl -s http://localhost:8307 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"dl-1","method":"load_csv_files","params":{"file_path":"data/sample.csv","encoding":"utf-8"}}' | jq
  ```
- **Pass Criteria**: returns row/column counts and preview.

### 3) Data Visualization (port 8308)
- **Endpoint**: `http://localhost:8308`
- **Validate**: Plotly figure spec
  ```bash
  curl -s http://localhost:8308 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"viz-1","method":"generate_plotly_chart","params":{"data_path":"data/sample.csv","x":"date","y":"sales"}}' | jq
  ```
- **Pass Criteria**: returns valid Plotly JSON spec.

### 4) Data Wrangling (port 8309)
- **Endpoint**: `http://localhost:8309`
- **Validate**: transform pipeline
  ```bash
  curl -s http://localhost:8309 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"wr-1","method":"transform_data","params":{"steps":[{"op":"dropna"},{"op":"astype","column":"price","to":"float"}]}}' | jq
  ```
- **Pass Criteria**: returns transformed preview & shape delta.

### 5) Feature Engineering (port 8310)
- **Endpoint**: `http://localhost:8310`
- **Validate**: feature selection
  ```bash
  curl -s http://localhost:8310 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"fe-1","method":"select_features","params":{"data_path":"data/sample.csv","target":"y"}}' | jq
  ```
- **Pass Criteria**: returns selected feature list & rationale.

### 6) SQL Database (port 8311)
- **Endpoint**: `http://localhost:8311`
- **Validate**: run SQL
  ```bash
  curl -s http://localhost:8311 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"sql-1","method":"execute_sql","params":{"conn":"postgres://user:pass@host:5432/db","query":"SELECT 1 AS ok"}}' | jq
  ```
- **Pass Criteria**: returns resultset schema & rows.

### 7) EDA Tools (port 8312)
- **Endpoint**: `http://localhost:8312`
- **Validate**: descriptive stats
  ```bash
  curl -s http://localhost:8312 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"eda-1","method":"compute_descriptive_stats","params":{"data_path":"data/sample.csv"}}' | jq
  ```

### 8) H2O ML (port 8313)
- **Endpoint**: `http://localhost:8313`
- **Validate**: AutoML
  ```bash
  curl -s http://localhost:8313 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"h2o-1","method":"train_automl","params":{"data_path":"data/sample.csv","target":"y","max_runtime_secs":30}}' | jq
  ```
- **Pass Criteria**: returns leaderboard & model key.

### 9) MLflow Tools (port 8314)
- **Endpoint**: `http://localhost:8314`
- **Validate**: log metrics
  ```bash
  curl -s http://localhost:8314 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"mlf-1","method":"log_metrics","params":{"run_name":"demo","metrics":{"rmse":0.12,"r2":0.88}}}' | jq
  ```

### 10) Pandas Analyst (port 8315 / alt 8210)
- **Endpoint**: `http://localhost:8315` (or `http://localhost:8210` if configured)
- **Validate**: inspect
  ```bash
  curl -s http://localhost:8315 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"pd-1","method":"inspect_data","params":{"file_path":"data/sample.csv","format":"csv"}}' | jq
  ```

### 11) Report Generator (port 8316)
- **Endpoint**: `http://localhost:8316`
- **Validate**: executive summary
  ```bash
  curl -s http://localhost:8316 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"rg-1","method":"generate_executive_summary","params":{"dataset_path":"data/sales.csv","business_goal":"분기 성과 요약"}}' | jq
  ```

---

## Deliverables
- All agents pass **healthcheck + 1 core capability** validation
- Langfuse v2 traces per agent run
- Playwright MCP E2E covers upload → analysis → report

## Notes
- Update `agents.json` to be the **single source of truth** for ports and endpoints.
- If any port differs in your environment, update the **Endpoint** lines above accordingly.