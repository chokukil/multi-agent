# Universal Engine API Reference

## 📋 개요

이 문서는 Universal Engine의 주요 API와 클래스들의 상세한 참조 자료를 제공합니다.

## 🎯 핵심 컴포넌트

### UniversalQueryProcessor

메인 쿼리 처리 엔진으로, 자연어 쿼리를 받아 LLM-First 방식으로 분석을 수행합니다.

#### 클래스 정의
```python
class UniversalQueryProcessor:
    """Universal Engine의 메인 쿼리 처리 클래스"""
```

#### 주요 메소드

##### `process_query()`
```python
async def process_query(
    self,
    query: str,
    data: Union[pd.DataFrame, Dict, str, None] = None,
    context: Dict[str, Any] = None
) -> Dict[str, Any]
```

**매개변수:**
- `query` (str): 자연어 쿼리 문자열
- `data` (Union[pd.DataFrame, Dict, str, None]): 처리할 데이터
- `context` (Dict[str, Any]): 추가 컨텍스트 정보

**반환값:**
- `Dict[str, Any]`: 분석 결과 딕셔너리

**예제:**
```python
processor = UniversalQueryProcessor()
result = await processor.process_query(
    query="매출 데이터의 월별 트렌드 분석",
    data=sales_df,
    context={"session_id": "sess123", "user_id": "user456"}
)
```

---

### MetaReasoningEngine

DeepSeek-R1에서 영감을 받은 4단계 메타추론 엔진입니다.

#### 클래스 정의
```python
class MetaReasoningEngine:
    """메타 추론 처리를 위한 엔진"""
    
    def __init__(self):
        self.reasoning_patterns = [
            "contextual_analysis",
            "logical_consistency_check", 
            "solution_generation",
            "quality_validation"
        ]
```

#### 주요 메소드

##### `analyze_request()`
```python
async def analyze_request(
    self,
    query: str,
    data: Any,
    context: Dict[str, Any]
) -> Dict[str, Any]
```

**매개변수:**
- `query` (str): 분석할 쿼리
- `data` (Any): 데이터 객체
- `context` (Dict[str, Any]): 컨텍스트 정보

**반환값:**
- `Dict[str, Any]`: 메타 추론 결과

**4단계 추론 과정:**
1. **컨텍스트 분석**: 쿼리와 데이터의 맥락 이해
2. **논리적 일관성 검사**: 요청의 논리적 타당성 검증
3. **솔루션 생성**: 최적의 분석 방법 결정
4. **품질 검증**: 결과의 품질과 신뢰성 평가

---

### DynamicContextDiscovery

동적 컨텍스트 발견 및 분석을 담당합니다.

#### 클래스 정의
```python
class DynamicContextDiscovery:
    """동적 컨텍스트 발견 시스템"""
    
    def __init__(self):
        self.discovered_contexts = {}
```

#### 주요 메소드

##### `discover_context()`
```python
async def discover_context(
    self,
    query: str,
    data: Any,
    existing_context: Dict[str, Any] = None
) -> Dict[str, Any]
```

**매개변수:**
- `query` (str): 컨텍스트를 발견할 쿼리
- `data` (Any): 분석 대상 데이터
- `existing_context` (Dict[str, Any]): 기존 컨텍스트 (선택사항)

**반환값:**
- `Dict[str, Any]`: 발견된 컨텍스트 정보

---

### AdaptiveUserUnderstanding

사용자 수준을 추정하고 맞춤형 응답을 제공합니다.

#### 클래스 정의
```python
class AdaptiveUserUnderstanding:
    """적응적 사용자 이해 시스템"""
    
    def __init__(self):
        self.user_models = {}
```

#### 주요 메소드

##### `estimate_user_level()`
```python
async def estimate_user_level(
    self,
    query: str,
    historical_queries: List[str] = None,
    user_feedback: Dict[str, Any] = None
) -> Dict[str, Any]
```

**매개변수:**
- `query` (str): 현재 사용자 쿼리
- `historical_queries` (List[str]): 과거 쿼리 목록
- `user_feedback` (Dict[str, Any]): 사용자 피드백

**반환값:**
- `Dict[str, Any]`: 추정된 사용자 프로필

**사용자 레벨:**
- `beginner`: 초보자 (기본적인 설명 필요)
- `intermediate`: 중급자 (적당한 기술적 세부사항)
- `expert`: 전문가 (고급 분석 및 상세한 통계)

---

### UniversalIntentDetection

쿼리의 의도를 분석하고 분류합니다.

#### 클래스 정의
```python
class UniversalIntentDetection:
    """범용 의도 탐지 시스템"""
    
    def __init__(self):
        self.intent_history = []
```

#### 주요 메소드

##### `detect_intent()`
```python
async def detect_intent(
    self,
    query: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]
```

**매개변수:**
- `query` (str): 의도를 분석할 쿼리
- `context` (Dict[str, Any]): 추가 컨텍스트

**반환값:**
- `Dict[str, Any]`: 탐지된 의도 정보

**의도 카테고리:**
- `data_analysis`: 데이터 분석
- `visualization`: 시각화
- `statistical_modeling`: 통계적 모델링
- `data_cleaning`: 데이터 전처리
- `reporting`: 보고서 생성

---

## 🔄 A2A 통합 시스템

### A2AAgentDiscoverySystem

Agent-to-Agent 시스템의 에이전트 발견을 담당합니다.

#### 클래스 정의
```python
class A2AAgentDiscoverySystem:
    """A2A 에이전트 발견 시스템"""
    
    def __init__(self):
        self.port_range = range(8306, 8316)  # 포트 8306-8315
```

#### 주요 메소드

##### `discover_agents()`
```python
async def discover_agents(self) -> Dict[str, Any]
```

**반환값:**
- `Dict[str, Any]`: 발견된 에이전트 정보

##### `check_agent_health()`
```python
async def check_agent_health(self) -> Dict[str, Any]
```

**반환값:**
- `Dict[str, Any]`: 각 에이전트의 상태 정보

**실제 에이전트 포트 매핑:**
- 8306: Data Cleaning Server (`a2a_ds_servers/data_cleaning_server.py`)
- 8307: Data Loader Server (`a2a_ds_servers/data_loader_server.py`)
- 8308: Data Visualization Server (`a2a_ds_servers/data_visualization_server.py`)
- 8309: Data Wrangling Server (`a2a_ds_servers/wrangling_server.py`)
- 8310: Feature Engineering Server (`a2a_ds_servers/feature_engineering_server.py`)
- 8311: SQL Data Analyst Server (`a2a_ds_servers/sql_data_analyst_server.py`)
- 8312: EDA Tools Server (`a2a_ds_servers/eda_tools_server.py`)
- 8313: H2O ML Server (`a2a_ds_servers/h2o_ml_server.py`)
- 8314: MLflow Server (`a2a_ds_servers/mlflow_server.py`)
- 8315: Report Generator Server (`a2a_ds_servers/report_generator_server.py`)

---

### A2AWorkflowOrchestrator

복잡한 워크플로우를 여러 에이전트에 분산하여 처리합니다.

#### 클래스 정의
```python
class A2AWorkflowOrchestrator:
    """A2A 워크플로우 오케스트레이터"""
```

#### 주요 메소드

##### `execute_workflow()`
```python
async def execute_workflow(
    self,
    query: str,
    data: Any,
    required_agents: List[str] = None
) -> Dict[str, Any]
```

**매개변수:**
- `query` (str): 실행할 워크플로우 쿼리
- `data` (Any): 처리할 데이터
- `required_agents` (List[str]): 필요한 에이전트 목록

**반환값:**
- `Dict[str, Any]`: 워크플로우 실행 결과

---

### A2AErrorHandler

A2A 시스템의 오류 처리 및 복구를 담당합니다.

#### 클래스 정의
```python
class A2AErrorHandler:
    """A2A 오류 처리 시스템"""
```

#### 주요 메소드

##### `handle_agent_error()`
```python
async def handle_agent_error(
    self,
    agent: Dict[str, Any],
    error: Exception,
    workflow_results: Dict[str, Any]
) -> Dict[str, Any]
```

**매개변수:**
- `agent` (Dict[str, Any]): 오류 발생 에이전트 정보
- `error` (Exception): 발생한 오류
- `workflow_results` (Dict[str, Any]): 현재 워크플로우 결과

**반환값:**
- `Dict[str, Any]`: 오류 처리 결과

---

## 🗂️ 세션 관리

### SessionManager

사용자 세션을 관리하고 컨텍스트를 유지합니다.

#### 클래스 정의
```python
class SessionManager:
    """세션 관리 시스템"""
```

#### 주요 메소드

##### `extract_comprehensive_context()`
```python
async def extract_comprehensive_context(
    self,
    session_data: Dict[str, Any]
) -> Dict[str, Any]
```

**매개변수:**
- `session_data` (Dict[str, Any]): 세션 데이터

**반환값:**
- `Dict[str, Any]`: 추출된 포괄적 컨텍스트

**세션 데이터 구조:**
```python
session_data = {
    "session_id": "unique_session_id",
    "user_id": "user_identifier", 
    "created_at": datetime.now(),
    "last_activity": datetime.now(),
    "messages": [
        {"role": "user", "content": "query"},
        {"role": "assistant", "content": "response"}
    ],
    "user_profile": {
        "expertise": "intermediate",
        "preferences": {"visualization": True}
    }
}
```

---

## 📊 모니터링 시스템

### PerformanceMonitoringSystem

시스템 성능을 모니터링하고 메트릭을 수집합니다.

#### 클래스 정의
```python
class PerformanceMonitoringSystem:
    """성능 모니터링 시스템"""
    
    def __init__(self):
        self.metrics_store = {}
        self.performance_thresholds = {
            "response_time": 5.0,  # 초
            "success_rate": 0.95,  # 95%
            "memory_usage": 0.80   # 80%
        }
```

#### 주요 메소드

##### `get_performance_metrics()`
```python
def get_performance_metrics(self) -> Dict[str, Any]
```

**반환값:**
- `Dict[str, Any]`: 성능 메트릭 정보

---

## 🏗️ 시스템 초기화

### UniversalEngineInitializer

Universal Engine의 전체 시스템을 초기화합니다.

#### 클래스 정의
```python
class UniversalEngineInitializer:
    """Universal Engine 시스템 초기화"""
```

#### 주요 메소드

##### `initialize_system()`
```python
async def initialize_system(self) -> bool
```

**반환값:**
- `bool`: 초기화 성공 여부

**초기화 단계:**
1. LLM Factory 설정
2. 핵심 엔진 컴포넌트 초기화
3. A2A 통합 시스템 설정
4. 세션 관리 시스템 준비
5. 모니터링 시스템 활성화

---

## 🔧 유틸리티 클래스

### LLMFactory

다양한 LLM 프로바이더와의 연동을 담당합니다.

#### 주요 메소드

##### `create_llm()`
```python
@staticmethod
def create_llm(provider: str = None) -> Any
```

**매개변수:**
- `provider` (str): LLM 제공업체 ("ollama", "openai" 등)

**반환값:**
- `Any`: 설정된 LLM 클라이언트

**지원 프로바이더:**
- `ollama`: 로컬 Ollama 서버
- `openai`: OpenAI API
- `anthropic`: Anthropic Claude
- `huggingface`: HuggingFace Transformers

---

## 📈 응답 형식

### 표준 응답 구조

모든 API 메소드는 다음 구조의 응답을 반환합니다:

```python
{
    "status": "success",  # success, error, warning
    "data": {
        # 실제 결과 데이터
        "analysis_result": "...",
        "confidence": 0.85,
        "reasoning": "..."
    },
    "metadata": {
        "processing_time": 2.5,  # 초
        "llm_calls": 3,
        "agents_used": ["data_cleaner", "eda_tools"],
        "user_level": "intermediate"
    },
    "errors": [],  # 오류 정보 (있는 경우)
    "warnings": []  # 경고 정보 (있는 경우)
}
```

### 오류 응답 구조

오류 발생 시 다음 구조로 응답합니다:

```python
{
    "status": "error",
    "error": {
        "type": "ValidationError",
        "message": "Invalid query format",
        "code": "E001",
        "details": {
            "query": "user_input",
            "expected_format": "string"
        }
    },
    "timestamp": "2025-01-22T10:30:00Z"
}
```

---

## 🔒 보안 고려사항

### 입력 검증

모든 사용자 입력은 다음과 같이 검증됩니다:

```python
# 악의적 패턴 차단
BLOCKED_PATTERNS = [
    r"'; DROP TABLE",  # SQL Injection
    r"<script.*?>",    # XSS
    r"\.\.\/",         # Path Traversal
    r"system\(",       # Command Injection
]
```

### 데이터 보안

민감한 데이터는 자동으로 마스킹됩니다:

```python
SENSITIVE_PATTERNS = {
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "credit_card": r"\d{4}-\d{4}-\d{4}-\d{4}",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
}
```

---

## 📚 타입 힌트

### 주요 타입 정의

```python
from typing import Any, Dict, List, Union, Optional
import pandas as pd

# 쿼리 타입
QueryType = str

# 데이터 타입
DataType = Union[pd.DataFrame, Dict[str, Any], str, None]

# 컨텍스트 타입
ContextType = Dict[str, Any]

# 결과 타입
ResultType = Dict[str, Any]

# 사용자 프로필 타입
UserProfileType = Dict[str, Union[str, int, float, bool, List]]

# 에이전트 정보 타입
AgentInfoType = Dict[str, Union[str, int, bool]]
```

---

## 🚀 성능 최적화

### 권장사항

1. **배치 처리**: 여러 쿼리를 함께 처리
2. **캐싱**: 반복되는 분석 결과 캐시 활용
3. **비동기 처리**: 모든 메소드는 async/await 사용
4. **커넥션 풀링**: A2A 에이전트 연결 재사용
5. **토큰 관리**: LLM 토큰 사용량 최적화

### 성능 메트릭

주요 성능 지표들:

- **응답 시간**: 평균 2-5초 목표
- **처리량**: 분당 100-500 쿼리
- **성공률**: 95% 이상 유지
- **메모리 사용량**: 80% 이하 유지
- **에이전트 가용성**: 90% 이상

---

이 API 참조 문서는 Universal Engine의 주요 컴포넌트와 사용법을 제공합니다. 추가적인 세부사항은 각 모듈의 소스 코드와 docstring을 참조하시기 바랍니다.