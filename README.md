[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chokukil/multi-agent)

# 🍒 Cherry AI - 데이터 사이언스 다중 에이전트 시스템

**Cherry AI**는 LangGraph의 Plan-Execute 패턴을 기반으로 구축된 강력한 데이터 사이언스 다중 에이전트 시스템입니다. Streamlit으로 구현된 직관적인 인터페이스를 통해 사용자는 데이터 분석, 시각화, 보고서 작성 등 복잡한 작업을 여러 전문 에이전트에게 위임하고 실시간으로 진행 상황을 추적할 수 있습니다.

## ✨ 주요 특징

-   **다중 에이전트 아키텍처**: 데이터 분석, 시각화 등 특정 역할에 특화된 여러 AI 에이전트가 협력하여 문제를 해결합니다.
-   **Plan-Execute 패턴**: `Planner`가 전체 작업 계획을 수립하면, `Router`가 각 단계를 가장 적합한 `Executor` 에이전트에게 동적으로 할당하여 체계적이고 효율적인 작업 수행을 보장합니다.
-   **SSOT (Single Source of Truth)**: 모든 에이전트가 `UnifiedDataManager`를 통해 동일한 데이터에 접근하여 데이터 불일치 문제를 방지하고 일관성을 유지합니다.
-   **데이터 계보 추적**: `DataLineageManager`를 통해 데이터의 모든 변환 과정을 추적하여 분석 결과의 투명성과 신뢰성을 확보합니다.
-   **동적인 시스템 구성**: 사용자는 UI를 통해 직접 에이전트를 추가/제거하고, 시스템 설정을 변경하며, 전체 구성을 템플릿으로 저장하고 불러올 수 있습니다.
-   **실시간 인터랙션 및 시각화**: Streamlit 기반의 인터랙티브한 UI를 통해 실시간으로 시스템과 소통하고, 분석 결과와 시스템 아키텍처를 시각적으로 확인할 수 있습니다.

## 🆕 A2A Protocol Standard Compliance

**Important Update**: CherryAI now fully complies with the A2A (Agent-to-Agent) protocol standard by using the official **a2a-sdk** package.

### A2A Implementation Details

- **Server**: Uses `A2AFastAPIApplication` with standard `DefaultRequestHandler`
- **Client**: Uses `A2AClient` with proper agent card discovery
- **Components**: Complete integration with `AgentExecutor`, `TaskStore`, and `RequestContext`
- **Standards**: Full compliance with A2A protocol specification v0.2.0+

### Key A2A Features

✅ **Standard Agent Card**: Served at `/.well-known/agent.json`  
✅ **Message Handling**: Complete A2A message protocol support  
✅ **Task Management**: Integrated with A2A task lifecycle  
✅ **Streaming**: Support for real-time communication  
✅ **Error Handling**: Proper A2A error responses  

### 🔬 A2A Data Science Agents

CherryAI includes a comprehensive suite of A2A-compliant data science agents:

#### Available Agents

| Agent | Port | Description |
|-------|------|-------------|
| **Data Loader** | 8000 | File operations, data loading, and preprocessing |
| **Pandas Analyst** | 8001 | Advanced pandas analysis with interactive visualizations |
| **SQL Analyst** | 8002 | Database queries and SQL-based analysis |
| **EDA Tools** | 8003 | Exploratory data analysis and statistical insights |
| **Data Visualization** | 8004 | Interactive charts and dashboard creation |
| **Orchestrator** | 8100 | Central management and coordination |

#### Key Features

🚀 **Real-time Streaming**: Live progress updates during analysis  
📊 **Interactive Visualizations**: Streamlit-optimized Plotly charts  
🔄 **Agent Orchestration**: Coordinated multi-agent workflows  
📁 **Artifact Management**: Automatic file and result storage  
🧪 **Sample Data**: Pre-loaded datasets for testing  

#### Usage

1. **Web Interface**: Visit the "🔬 A2A Data Science" page in CherryAI
2. **Agent Management**: Monitor and control agents via "⚙️ Agent Management"
3. **Direct API**: Send A2A protocol requests to individual agents
4. **System Control**: Use `system_start.bat` to launch all agents

### A2A Architecture
