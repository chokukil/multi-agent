# Universal Engine User Guide

## 🚀 개요

Universal Engine은 CherryAI의 핵심 지능형 시스템으로, 제로 하드코딩 철학을 바탕으로 구축된 LLM-First 아키텍처입니다. DeepSeek-R1에서 영감을 받은 4단계 메타추론 프로세스를 통해 사용자의 다양한 데이터 분석 요청에 적응적으로 대응합니다.

## 🎯 주요 특징

### 1. LLM-First 아키텍처
- **제로 하드코딩**: 모든 로직이 LLM을 통해 동적으로 생성
- **적응적 추론**: 각 요청에 맞는 최적의 처리 방식 자동 결정
- **메타 추론**: 4단계 추론 과정으로 복잡한 문제 해결

### 2. Universal Query Processing
- **자연어 쿼리 처리**: 일반 언어로 데이터 분석 요청
- **다양한 데이터 형식 지원**: CSV, JSON, DataFrame 등
- **컨텍스트 인식**: 사용자 프로필과 과거 상호작용 기반 맞춤 응답

### 3. A2A (Agent-to-Agent) 통합
- **10개 전문 에이전트**: 포트 8306-8315에서 운영
  - 8306: 데이터 정리, 8307: 데이터 로더, 8308: 시각화, 8309: 데이터 가공
  - 8310: 피처 엔지니어링, 8311: SQL 분석, 8312: EDA 도구, 8313: H2O ML
  - 8314: MLflow 도구, 8315: 보고서 생성
- **자동 에이전트 발견**: 동적 에이전트 탐지 및 할당
- **워크플로우 오케스트레이션**: 복잡한 작업의 자동 분산 처리

### 4. 적응적 사용자 이해
- **사용자 레벨 추정**: 초보자부터 전문가까지 자동 판별
- **Progressive Disclosure**: 사용자 수준에 맞는 정보 제공
- **실시간 학습**: 사용자 상호작용을 통한 지속적 개선

## 📋 시스템 요구사항

### 최소 요구사항
- **Python**: 3.9 이상
- **메모리**: 4GB RAM 이상
- **디스크**: 2GB 여유 공간
- **LLM Provider**: Ollama, OpenAI, 또는 기타 지원 모델

### 권장 요구사항
- **Python**: 3.11 이상
- **메모리**: 8GB RAM 이상
- **CPU**: 4코어 이상
- **GPU**: CUDA 지원 (선택사항, 성능 향상)

## 🛠 설치 및 설정

### 1. 환경 변수 설정

```bash
# LLM Provider 설정
export LLM_PROVIDER="ollama"  # 또는 "openai"
export OLLAMA_MODEL="llama2"  # Ollama 사용 시
export OPENAI_API_KEY="your-key"  # OpenAI 사용 시

# A2A 에이전트 설정
export A2A_PORT_START="8306"
export A2A_PORT_END="8315"
```

### 2. 의존성 설치

```bash
# 가상환경 활성화
source .venv/bin/activate

# 필수 패키지 설치
pip install -r requirements.txt
```

### 3. 시스템 초기화

```python
from core.universal_engine.initialization.system_initializer import UniversalEngineInitializer

# 시스템 초기화
initializer = UniversalEngineInitializer()
await initializer.initialize_system()
```

## 🚀 기본 사용법

### 1. 간단한 쿼리 처리

```python
from core.universal_engine.universal_query_processor import UniversalQueryProcessor
import pandas as pd

# 데이터 준비
data = pd.read_csv("your_data.csv")

# Query Processor 초기화
processor = UniversalQueryProcessor()

# 자연어 쿼리 실행
result = await processor.process_query(
    query="이 데이터의 주요 트렌드를 분석해주세요",
    data=data,
    context={"session_id": "user123"}
)

print(result)
```

### 2. 세션 관리

```python
from core.universal_engine.session.session_management_system import SessionManager

# 세션 매니저 초기화
session_manager = SessionManager()

# 세션 생성
session = {
    "session_id": "unique_session_123",
    "user_id": "user456",
    "messages": [],
    "user_profile": {"expertise": "intermediate"}
}

# 컨텍스트 추출
context = await session_manager.extract_comprehensive_context(session)
```

### 3. A2A 에이전트 활용

```python
from core.universal_engine.a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator

# 워크플로우 오케스트레이터 초기화
orchestrator = A2AWorkflowOrchestrator()

# 복잡한 분석 작업 실행
workflow_result = await orchestrator.execute_workflow(
    query="데이터 정리 후 고급 통계 분석 수행",
    data=large_dataset,
    required_agents=["data_cleaner", "statistical_analyst"]
)
```

## 🎛 고급 설정

### 1. 사용자 레벨 커스터마이징

```python
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding

# 사용자 이해 모듈 초기화
user_understanding = AdaptiveUserUnderstanding()

# 사용자 프로필 설정
user_profile = await user_understanding.estimate_user_level(
    query="복잡한 머신러닝 모델링을 위한 특성 엔지니어링",
    historical_queries=["회귀분석", "교차검증", "하이퍼파라미터 튜닝"],
    user_feedback={"complexity_preference": "high"}
)
```

### 2. 동적 컨텍스트 발견

```python
from core.universal_engine.dynamic_context_discovery import DynamicContextDiscovery

# 컨텍스트 발견 모듈 초기화
context_discovery = DynamicContextDiscovery()

# 동적 컨텍스트 분석
discovered_context = await context_discovery.discover_context(
    query="고객 이탈 예측 모델",
    data=customer_data,
    domain_knowledge={"industry": "telecommunications"}
)
```

### 3. 메타 추론 엔진 활용

```python
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine

# 메타 추론 엔진 초기화
meta_engine = MetaReasoningEngine()

# 4단계 추론 프로세스 실행
reasoning_result = await meta_engine.analyze_request(
    query="시계열 데이터의 이상치 탐지 및 예측",
    data=time_series_data,
    context={"domain": "finance", "urgency": "high"}
)
```

## 📊 모니터링 및 성능

### 1. 성능 모니터링

```python
from core.universal_engine.monitoring.performance_monitoring_system import PerformanceMonitoringSystem

# 모니터링 시스템 초기화
monitor = PerformanceMonitoringSystem()

# 성능 메트릭 확인
metrics = monitor.get_performance_metrics()
print(f"평균 응답시간: {metrics['avg_response_time']:.2f}초")
print(f"성공률: {metrics['success_rate']:.1f}%")
```

### 2. 시스템 상태 확인

```python
# A2A 에이전트 상태 확인
from core.universal_engine.a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem

discovery = A2AAgentDiscoverySystem()
agent_status = await discovery.check_agent_health()

for agent_id, status in agent_status.items():
    print(f"에이전트 {agent_id}: {status['status']} (포트: {status['port']})")
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. LLM 연결 오류
```
Error: LLM service unavailable
```

**해결 방법:**
1. LLM_PROVIDER 환경변수 확인
2. Ollama 서비스 실행 상태 확인: `ollama serve`
3. OpenAI API 키 유효성 검증

#### 2. A2A 에이전트 연결 실패
```
Error: A2A agents not responding
```

**해결 방법:**
1. 포트 범위 확인 (8306-8315)
2. 방화벽 설정 검토
3. 에이전트 서비스 재시작

#### 3. 메모리 부족 오류
```
Error: Out of memory
```

**해결 방법:**
1. 대용량 데이터 청크 단위 처리
2. 시스템 메모리 증설
3. 배치 크기 조정

### 로그 확인

```python
import logging

# 로그 레벨 설정
logging.basicConfig(level=logging.INFO)

# Universal Engine 로그 확인
logger = logging.getLogger('universal_engine')
logger.info("시스템 상태 확인")
```

## 📈 성능 최적화 가이드

### 1. 데이터 처리 최적화
- **청크 단위 처리**: 대용량 데이터는 작은 단위로 분할
- **캐싱 활용**: 반복적인 쿼리 결과 캐시 저장
- **인덱싱**: 자주 사용하는 컬럼에 인덱스 생성

### 2. LLM 호출 최적화
- **배치 처리**: 여러 요청을 묶어서 처리
- **프롬프트 최적화**: 간결하고 명확한 프롬프트 작성
- **토큰 제한**: 입력 토큰 수 관리

### 3. A2A 에이전트 최적화
- **로드 밸런싱**: 에이전트 간 작업 분산
- **Connection Pooling**: 연결 재사용
- **Circuit Breaker**: 장애 에이전트 자동 차단

## 🔒 보안 고려사항

### 1. 데이터 보안
- **민감 정보 마스킹**: SSN, 신용카드 번호 등 자동 마스킹
- **접근 권한 관리**: 사용자별 데이터 접근 제한
- **암호화**: 저장 및 전송 중 데이터 암호화

### 2. 세션 보안
- **세션 만료**: 비활성 세션 자동 정리
- **세션 격리**: 사용자 간 데이터 분리
- **보안 헤더**: 적절한 HTTP 보안 헤더 설정

### 3. 입력 검증
- **SQL 인젝션 방지**: 악의적 쿼리 차단
- **XSS 방지**: 스크립트 인젝션 방지
- **입력 크기 제한**: 과도한 입력 차단

## 🎯 사용 사례

### 1. 비즈니스 분석가
```python
# 매출 트렌드 분석
result = await processor.process_query(
    query="지난 6개월 매출 트렌드를 분석하고 다음 분기 예측을 해주세요",
    data=sales_data,
    context={"user_role": "analyst", "expertise": "intermediate"}
)
```

### 2. 데이터 과학자
```python
# 고급 머신러닝 분석
result = await processor.process_query(
    query="고객 세그먼테이션을 위한 클러스터링 분석 후 각 세그먼트의 특성 분석",
    data=customer_data,
    context={"user_role": "data_scientist", "expertise": "expert"}
)
```

### 3. 비개발자
```python
# 간단한 데이터 요약
result = await processor.process_query(
    query="이 데이터에서 중요한 것들을 간단히 설명해주세요",
    data=survey_data,
    context={"user_role": "business_user", "expertise": "beginner"}
)
```

## 📚 추가 리소스

- **API 참조**: [UNIVERSAL_ENGINE_API_REFERENCE.md](./UNIVERSAL_ENGINE_API_REFERENCE.md)
- **배포 가이드**: [UNIVERSAL_ENGINE_DEPLOYMENT_GUIDE.md](./UNIVERSAL_ENGINE_DEPLOYMENT_GUIDE.md)
- **문제 해결**: [UNIVERSAL_ENGINE_TROUBLESHOOTING.md](./UNIVERSAL_ENGINE_TROUBLESHOOTING.md)
- **전체 시스템 가이드**: [USER_GUIDE.md](./USER_GUIDE.md)

## 🆘 지원 및 커뮤니티

문제가 발생하거나 질문이 있으시면:

1. **문제 해결 가이드** 확인
2. **로그 파일** 분석
3. **GitHub Issues**에 문제 보고
4. **커뮤니티 포럼** 참여

---

**Universal Engine v1.0** - LLM-First 데이터 분석의 새로운 패러다임