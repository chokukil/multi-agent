# 🍒 Cherry AI - 데이터 사이언스 다중 에이전트 시스템

**Cherry AI**는 LangGraph의 Plan-Execute 패턴을 기반으로 구축된 강력한 데이터 사이언스 다중 에이전트 시스템입니다. Streamlit으로 구현된 직관적인 인터페이스를 통해 사용자는 데이터 분석, 시각화, 보고서 작성 등 복잡한 작업을 여러 전문 에이전트에게 위임하고 실시간으로 진행 상황을 추적할 수 있습니다.

## ✨ 주요 특징

-   **다중 에이전트 아키텍처**: 데이터 분석, 시각화 등 특정 역할에 특화된 여러 AI 에이전트가 협력하여 문제를 해결합니다.
-   **Plan-Execute 패턴**: `Planner`가 전체 작업 계획을 수립하면, `Router`가 각 단계를 가장 적합한 `Executor` 에이전트에게 동적으로 할당하여 체계적이고 효율적인 작업 수행을 보장합니다.
-   **SSOT (Single Source of Truth)**: 모든 에이전트가 `UnifiedDataManager`를 통해 동일한 데이터에 접근하여 데이터 불일치 문제를 방지하고 일관성을 유지합니다.
-   **데이터 계보 추적**: `DataLineageManager`를 통해 데이터의 모든 변환 과정을 추적하여 분석 결과의 투명성과 신뢰성을 확보합니다.
-   **동적인 시스템 구성**: 사용자는 UI를 통해 직접 에이전트를 추가/제거하고, 시스템 설정을 변경하며, 전체 구성을 템플릿으로 저장하고 불러올 수 있습니다.
-   **실시간 인터랙션 및 시각화**: Streamlit 기반의 인터랙티브한 UI를 통해 실시간으로 시스템과 소통하고, 분석 결과와 시스템 아키텍처를 시각적으로 확인할 수 있습니다.

## 🏛️ 프로젝트 구조

```
CherryAI/
├── app.py                   # Streamlit 애플리케이션의 메인 진입점
├── core/                    # 핵심 비즈니스 로직
│   ├── data_manager.py      # 데이터 관리 (SSOT)
│   ├── llm_factory.py       # LLM 인스턴스 생성
│   ├── plan_execute/        # Plan-Execute 패턴 구현
│   │   ├── planner.py       # 작업 계획 수립
│   │   ├── router.py        # 작업 라우팅
│   │   └── executor.py      # 실제 작업 수행
│   └── tools/               # 에이전트가 사용하는 도구 (e.g., Python REPL)
├── ui/                      # Streamlit UI 컴포넌트
│   ├── chat_interface.py    # 채팅 인터페이스
│   └── sidebar_components.py # 사이드바 UI 요소
├── mcp-servers/             # MCP(Multi-Agent Control Plane) 서버 스크립트 (선택 사항)
├── multi_agent_supervisor.py# 다중 에이전트 시스템 관리자 UI
├── pyproject.toml           # 프로젝트 의존성 관리 (uv)
└── requirements.txt         # 프로젝트 의존성 관리 (pip)
```

## 🚀 시작하기

### 1. 환경 설정

저장소를 클론하고 프로젝트 디렉토리로 이동합니다.

```bash
git clone <repository-url>
cd CherryAI_0621
```

### 2. 의존성 설치

`uv` 패키지 매니저를 사용하여 의존성을 설치합니다.

```bash
# uv 사용 (권장)
uv sync

# 또는 pip 사용
pip install -r requirements.txt
```

### 3. LLM 제공자 설정

#### Option A: OpenAI (유료)
```bash
cp .env.example .env
# .env 파일에 API 키 설정
OPENAI_API_KEY="sk-..."
LLM_PROVIDER="OPENAI"
```

#### Option B: Ollama (무료, 로컬)
```bash
# 🦙 Ollama 자동 설정 (권장)
./setup_ollama_env.sh      # Linux/macOS
# setup_ollama_env.bat     # Windows

# 또는 수동 설정
export LLM_PROVIDER=OLLAMA
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434

# Ollama 서버 시작
ollama serve

# 권장 모델 다운로드
ollama pull llama3.1:8b
```

#### 🆕 Ollama Tool Calling 지원
Cherry AI는 Ollama의 도구 호출을 완전히 지원합니다:
- ✅ **GPT와 동일한 성능**: 데이터 분석, 시각화, 코드 실행
- ✅ **자동 모델 검증**: 도구 호출 지원 모델 자동 감지
- ✅ **최적화된 에이전트**: Ollama 전용 커스텀 에이전트
- ✅ **실시간 모니터링**: UI에서 Ollama 상태 확인

**권장 모델**:
- `llama3.1:8b` - 균형잡힌 성능 (10GB RAM)
- `qwen2.5:7b` - 빠른 처리 (8GB RAM)
- `qwen2.5-coder:7b` - 코딩 전문 (9GB RAM)

자세한 설정은 [OLLAMA_IMPROVEMENT_GUIDE.md](./OLLAMA_IMPROVEMENT_GUIDE.md)를 참조하세요.

### 4. 애플리케이션 실행

다음 명령어를 사용하여 Streamlit 애플리케이션을 실행합니다.

```bash
streamlit run app.py
```

브라우저에서 애플리케이션이 열리면, 사이드바의 "Quick Start" 템플릿을 사용하거나 직접 에이전트를 구성하여 데이터 분석을 시작할 수 있습니다.
