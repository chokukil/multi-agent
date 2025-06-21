# CherryAI 시스템 시작 가이드

## 🚀 빠른 시작

### 1단계: 의존성 설치 (최초 1회)
```batch
install_mcp_dependencies.bat
```

### 2단계: 시스템 시작
```batch
system_start.bat
```
또는 빠른 시작:
```batch
quick_start.bat
```

### 3단계: 시스템 종료
```batch
system_stop.bat
```

---

## 📋 배치 파일 설명

### `install_mcp_dependencies.bat`
- **목적**: MCP 서버에 필요한 모든 패키지를 단계별로 설치
- **언제 사용**: 최초 설치시 또는 의존성 문제 발생시
- **특징**: 
  - 핵심 패키지와 선택적 패키지를 구분하여 설치
  - 설치 실패시에도 다음 단계 진행
  - 상세한 설치 로그 제공

### `system_start.bat`
- **목적**: 완전한 시스템 시작 (MCP 서버 + Streamlit 앱)
- **동작**: 
  1. MCP 서버들 시작
  2. 45초 대기 (안정화)
  3. Streamlit 앱 실행
- **특징**: 안전하고 확실한 시작, 상세한 상태 메시지

### `quick_start.bat`
- **목적**: 빠른 시스템 시작
- **동작**:
  1. MCP 서버들을 백그라운드에서 시작
  2. 15초 대기 후 즉시 Streamlit 실행
- **특징**: 빠른 시작, 간소화된 프로세스

### `mcp_server_start.bat`
- **목적**: MCP 서버들만 개별적으로 시작
- **특징**: 각 서버를 별도 창에서 실행하여 개별 모니터링 가능

### `system_stop.bat`
- **목적**: 모든 CherryAI 서비스 안전 종료
- **동작**: Streamlit, MCP 서버, 관련 Python 프로세스 모두 종료

---

## 🔧 시스템 구성

### MCP 서버 목록 (포트)
- **파일 관리**: localhost:8006
- **데이터 사이언스**: localhost:8007  
- **반도체 수율 분석**: localhost:8008
- **공정 제어**: localhost:8009
- **장비 분석**: localhost:8010
- **결함 패턴 분석**: localhost:8011
- **공정 최적화**: localhost:8012
- **시계열 분석**: localhost:8013
- **이상 탐지**: localhost:8014
- **고급 ML**: localhost:8016
- **데이터 전처리**: localhost:8017
- **통계 분석**: localhost:8018
- **보고서 작성**: localhost:8019
- **반도체 공정**: localhost:8020

### 메인 애플리케이션
- **Streamlit 앱**: localhost:8501

---

## 🛠️ 문제 해결

### 의존성 오류가 발생할 때
```batch
install_mcp_dependencies.bat
```

### 포트 충돌이 발생할 때
1. `system_stop.bat` 실행
2. 몇 초 대기 후 다시 시작

### 서버가 시작되지 않을 때
1. `.venv` 폴더가 있는지 확인
2. `uv --version`으로 UV 설치 확인
3. 수동으로 개별 서버 실행해서 오류 확인

### 완전 초기화가 필요할 때
1. `.venv` 폴더 삭제
2. `install_mcp_dependencies.bat` 다시 실행

---

## 💡 사용 팁

1. **첫 실행**: `install_mcp_dependencies.bat` → `system_start.bat`
2. **일반 사용**: `quick_start.bat`
3. **개발/디버깅**: `mcp_server_start.bat` → 수동으로 Streamlit 실행
4. **종료**: `system_stop.bat` 또는 Ctrl+C

---

## 📦 설치된 주요 패키지

### 핵심 패키지
- FastMCP, MCP, Uvicorn (MCP 프레임워크)
- Pandas, NumPy, Matplotlib, Seaborn (데이터 처리)
- Scikit-learn, XGBoost (기본 ML)

### 고급 ML
- CatBoost, LightGBM (그래디언트 부스팅)
- SHAP, LIME (모델 해석)
- Optuna (하이퍼파라미터 최적화)

### 통계 분석
- Statsmodels, Pingouin (통계)
- UMAP (차원 축소)
- Imbalanced-learn (불균형 데이터) 