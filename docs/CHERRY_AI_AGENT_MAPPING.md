# CherryAI 에이전트 매핑 가이드

## 📂 전체 디렉토리 구조

- **`a2a_ds_servers/`**: 데이터 과학 관련 A2A 에이전트 서버 파일들이 위치합니다.
- **`scripts/`**: 시스템 시작/종료 등 유틸리티 스크립트가 위치합니다.
- **`docs/`**: 시스템 관련 문서들이 위치합니다.
- **`monitoring/`**: 모니터링 관련 설정 파일이 위치합니다.

## 🗺️ 포트별 서버 매핑 테이블 (업데이트: 2025-01-23)

| 포트 | 에이전트 | 서버 파일 (구현) | 새 구현 | 상태 | 테스트 성공률 |
|---|---|---|---|---|---|
| 8306 | Data Cleaning | `data_cleaning_server.py` | ✅ `data_cleaning_server_new.py` | **완료** | 100% |
| 8308 | Data Visualization | `data_visualization_server.py` | ✅ `data_visualization_server_new.py` | **완료** | 100% |
| 8309 | Data Wrangling | `wrangling_server.py` | ✅ `data_wrangling_server_new.py` | **완료** | 100% |
| 8310 | Feature Engineering | `feature_engineering_server.py` | ✅ `feature_engineering_server_new.py` | **완료** | 100% |
| 8312 | EDA Tools | `eda_tools_server.py` | ⏳ `eda_tools_server_new.py` | **Phase 3 예정** | - |
| 8313 | H2O ML | `h2o_ml_server.py` | ⏳ `h2o_ml_server_new.py` | **Phase 4 예정** | - |
| 8311 | SQL Data Analyst | `sql_data_analyst_server.py` | ⏳ `sql_data_analyst_server_new.py` | **Phase 5 예정** | - |
| 8314 | MLflow | `mlflow_server.py` | ⏳ `mlflow_server_new.py` | **Phase 6 예정** | - |
| 8315 | Data Loader | `data_loader_server.py` | ⏳ `data_loader_server_new.py` | **Phase 7 예정** | - |
| 8316 | Pandas Analyst | `pandas_data_analyst_server.py` | ⏳ `pandas_analyst_server_new.py` | **Phase 8 예정** | - |
| 8317 | Report Generator | `report_generator_server.py` | ✅ `report_generator_server_new.py` | **Phase 9 예정** | - |

### 🎯 **진행 현황 요약**
- **완료**: 4개 에이전트 (36%)
- **남은 작업**: 7개 에이전트 (64%)
- **총 기능**: 88개 중 32개 완료 (36%)

## 🚀 실제 시작 명령어들

### 전체 시스템 시작
```bash
./scripts/start_universal_engine.sh
```

### 완료된 새 에이전트 시작 (A2A SDK 0.2.9)
```bash
# Phase 0: Data Cleaning Agent (포트 8306)
python a2a_ds_servers/data_cleaning_server_new.py

# Phase 1: Data Visualization Agent (포트 8308)  
python a2a_ds_servers/data_visualization_server_new.py

# Phase 1: Data Wrangling Agent (포트 8309)
python a2a_ds_servers/data_wrangling_server_new.py

# Phase 2: Feature Engineering Agent (포트 8310)
python a2a_ds_servers/feature_engineering_server_new.py
```

### 기존 에이전트 시작 (호환성)
```bash
# 기존 구현들 (TaskUpdater 패턴 미적용)
python a2a_ds_servers/data_cleaning_server.py
python a2a_ds_servers/data_visualization_server.py  
python a2a_ds_servers/wrangling_server.py
python a2a_ds_servers/feature_engineering_server.py
```

## ✅ 헬스 체크 및 관리 방법

### 전체 시스템 헬스 체크
```bash
curl http://localhost:8000/health
```

### 개별 에이전트 헬스 체크 (예: Data Cleaning Agent)
```bash
curl http://localhost:8306/health
```

## 💻 개발 및 확장 가이드

새로운 에이전트를 추가하려면 다음 단계를 따르세요.

1.  **서버 파일 생성**: `a2a_ds_servers/` 디렉토리에 새로운 에이전트 서버 파일(e.g., `new_agent_server.py`)을 생성합니다.
2.  **포트 할당**: 사용하지 않는 포트를 새로운 에이전트에 할당합니다.
3.  **스크립트 업데이트**: `scripts/start_universal_engine.sh` 파일의 `start_a2a_agents` 함수에 새로운 에이전트 정보를 추가합니다.
4.  **문서 업데이트**: 이 매핑 가이드(`CHERRY_AI_AGENT_MAPPING.md`)와 관련 문서를 업데이트합니다.
5.  **모니터링 설정**: `monitoring/prometheus.yml`에 새로운 에이전트의 메트릭 수집 설정을 추가합니다.
