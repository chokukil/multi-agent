# AI Data Science Team A2A Wrapper Migration Plan

## 📋 개요

AI Data Science Team 라이브러리를 A2A SDK v0.2.9와 완전히 호환되도록 래핑하는 체계적인 전환 계획입니다.

## 🎯 목표

1. **표준화**: 모든 AI DS Team 에이전트를 A2A 프로토콜 표준에 맞게 통합
2. **안정성**: `get_workflow_summary` 등의 호환성 문제 해결
3. **확장성**: 새로운 에이전트 추가 시 일관된 패턴 제공
4. **성능**: 스트리밍 및 비동기 처리 최적화

## 🏗️ 아키텍처 설계

### Base Wrapper 구조
```
a2a_ds_servers/base/
├── __init__.py                 # 패키지 초기화
├── ai_ds_team_wrapper.py      # 기본 래퍼 클래스
├── streaming_wrapper.py       # 스트리밍 래퍼 클래스
└── utils.py                   # 공통 유틸리티
```

### 핵심 컴포넌트

1. **AIDataScienceTeamWrapper**: 기본 A2A 래퍼 클래스
   - AgentExecutor 상속
   - 표준 execute/cancel 메서드 구현
   - 안전한 에이전트 실행 및 오류 처리

2. **StreamingAIDataScienceWrapper**: 스트리밍 지원 래퍼
   - 네이티브 스트리밍 지원 (astream/stream)
   - 시뮬레이션 스트리밍 (비지원 에이전트용)
   - 청크 단위 데이터 전송

3. **Utilities**: 공통 유틸리티 함수들
   - `extract_user_input`: A2A 컨텍스트에서 사용자 입력 추출
   - `safe_get_workflow_summary`: 안전한 워크플로우 요약 가져오기
   - `convert_ai_ds_response_to_a2a`: AI DS Team 응답을 A2A 형식으로 변환

## 📊 현재 서버 현황 분석

### 기존 서버들 (전환 필요)

| 서버명 | 포트 | 상태 | 주요 문제 | 우선순위 |
|--------|------|------|-----------|----------|
| Orchestrator | 8100 | 🟢 작동 | 계획 생성만, 실행 안됨 | 🔴 High |
| Data Loader | 8307 | 🔴 오류 | get_workflow_summary | 🔴 High |
| Data Cleaning | 8306 | 🟡 부분작동 | 안전처리 구현됨 | 🟡 Medium |
| EDA Tools | 8312 | 🔴 오류 | get_workflow_summary | 🔴 High |
| Data Visualization | 8313 | 🔴 오류 | get_workflow_summary | 🔴 High |
| Feature Engineering | 8314 | 🔴 오류 | get_workflow_summary | 🔴 High |
| H2O Modeling | 8315 | 🔴 오류 | get_workflow_summary | 🟡 Medium |
| MLflow | 8316 | 🔴 오류 | get_workflow_summary | 🟡 Medium |
| SQL Database | 8317 | 🔴 오류 | get_workflow_summary | 🟡 Medium |
| Data Wrangling | 8318 | 🔴 오류 | get_workflow_summary | 🔴 High |

## 🔄 전환 단계별 계획

### Phase 1: 베이스 래퍼 구현 ✅
- [x] AIDataScienceTeamWrapper 기본 클래스
- [x] StreamingAIDataScienceWrapper 스트리밍 클래스
- [x] 공통 유틸리티 함수들
- [x] 패키지 구조 설정

### Phase 2: 핵심 서버 전환 (진행 중)
#### 2.1 Data Loader (Port 8307) 🔴
- [ ] 기존 서버 분석
- [ ] 새로운 래퍼로 전환
- [ ] 데이터 로딩 특화 기능 구현
- [ ] 테스트 및 검증

#### 2.2 Data Cleaning (Port 8306) 🟡
- [x] 기존 안전 처리 확인
- [ ] 새로운 래퍼로 개선
- [ ] 데이터 정리 특화 기능 강화
- [ ] 테스트 및 검증

#### 2.3 EDA Tools (Port 8312) 🔴
- [ ] 기존 서버 분석
- [ ] 새로운 래퍼로 전환
- [ ] EDA 아티팩트 수집 기능
- [ ] 테스트 및 검증

### Phase 3: 시각화 및 특성 엔지니어링 서버
#### 3.1 Data Visualization (Port 8313)
- [ ] 시각화 특화 래퍼 구현
- [ ] 차트 생성 및 저장 기능
- [ ] 이미지 아티팩트 처리

#### 3.2 Feature Engineering (Port 8314)
- [ ] 특성 엔지니어링 특화 래퍼
- [ ] 변환된 데이터 관리
- [ ] 특성 중요도 분석

### Phase 4: ML 관련 서버
#### 4.1 H2O Modeling (Port 8315)
- [ ] H2O 모델링 래퍼
- [ ] 모델 아티팩트 관리
- [ ] 성능 메트릭 처리

#### 4.2 MLflow (Port 8316)
- [ ] MLflow 실험 관리
- [ ] 모델 버전 관리
- [ ] 메트릭 추적

### Phase 5: 데이터베이스 및 고급 기능
#### 5.1 SQL Database (Port 8317)
- [ ] SQL 쿼리 실행 래퍼
- [ ] 데이터베이스 연결 관리
- [ ] 쿼리 결과 처리

#### 5.2 Data Wrangling (Port 8318)
- [ ] 데이터 랭글링 특화 기능
- [ ] 복잡한 데이터 변환
- [ ] 파이프라인 관리

### Phase 6: 통합 테스트 및 최적화
- [ ] 전체 시스템 통합 테스트
- [ ] 오케스트레이터와의 연동 확인
- [ ] 성능 최적화
- [ ] 문서화 완료

## 🧪 테스트 전략

### 단위 테스트
```python
# 각 래퍼 클래스별 테스트
test_data_loader_wrapper()
test_data_cleaning_wrapper()
test_eda_tools_wrapper()
# ... 기타 에이전트들
```

### 통합 테스트
```python
# A2A 프로토콜 호환성 테스트
test_a2a_protocol_compatibility()
test_orchestrator_integration()
test_streaming_functionality()
```

### 시스템 테스트
```python
# 전체 워크플로우 테스트
test_complete_eda_workflow()
test_data_pipeline_execution()
test_ml_model_training_pipeline()
```

## 🔧 구현 가이드라인

### 1. 에이전트별 래퍼 구현 패턴

```python
class SpecificAgentWrapper(AIDataScienceTeamWrapper):
    def __init__(self):
        # LLM 설정
        llm = create_llm_instance()
        
        # 에이전트별 특화 설정
        agent_config = {
            "model": llm,
            # 에이전트별 추가 설정
        }
        
        super().__init__(
            agent_class=SpecificAgent,
            agent_config=agent_config,
            agent_name="Specific Agent Name"
        )
    
    async def _execute_agent(self, user_input: str) -> any:
        # 에이전트별 특화 실행 로직
        pass
    
    def _build_final_response(self, workflow_summary: str, a2a_response: dict, user_input: str) -> str:
        # 에이전트별 특화 응답 구성
        pass
```

### 2. 오류 처리 표준

```python
try:
    # 에이전트 실행
    result = self.agent.invoke_agent(user_input, data)
    
    # 워크플로우 요약 안전 가져오기
    workflow_summary = safe_get_workflow_summary(
        self.agent, 
        f"✅ {self.agent_name} 작업이 완료되었습니다."
    )
    
except Exception as e:
    logger.error(f"Agent execution failed: {e}")
    return create_error_response(str(e), self.agent_name)
```

### 3. 아티팩트 관리

```python
def _collect_artifacts(self) -> dict:
    artifacts = {
        "reports": [],
        "plots": [],
        "data": []
    }
    
    # 각 아티팩트 타입별 수집 로직
    return artifacts
```

## 📈 성공 지표

### 기능적 지표
- [ ] 모든 에이전트 정상 작동 (100%)
- [ ] get_workflow_summary 오류 해결 (100%)
- [ ] A2A 프로토콜 완전 호환 (100%)
- [ ] 오케스트레이터 연동 성공 (100%)

### 성능 지표
- [ ] 응답 시간 < 30초 (95th percentile)
- [ ] 메모리 사용량 최적화
- [ ] 에러율 < 1%
- [ ] 스트리밍 지연 < 1초

### 사용성 지표
- [ ] 일관된 응답 형식
- [ ] 명확한 오류 메시지
- [ ] 상세한 진행 상황 표시
- [ ] 아티팩트 자동 관리

## 🚀 배포 계획

### 1. 점진적 배포
- 기존 서버와 새 서버 병렬 운영
- 포트 번호 변경으로 구분 (예: 8306 → 8406)
- 안정성 확인 후 포트 교체

### 2. 롤백 계획
- 기존 서버 백업 유지
- 문제 발생 시 즉시 이전 버전으로 복원
- 설정 파일 버전 관리

### 3. 모니터링
- 서버 상태 실시간 모니터링
- 에러 로그 중앙 집중 관리
- 성능 메트릭 추적

## 📚 참고 자료

- [A2A SDK v0.2.9 공식 문서](https://github.com/a2aproject/a2a-python)
- [AI Data Science Team 라이브러리](https://github.com/business-science/ai-data-science-team)
- [LangGraph 스트리밍 가이드](https://langchain-ai.github.io/langgraph/)

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 | 담당자 |
|------|------|-----------|--------|
| 2025-01-20 | 1.0.0 | 초기 전환 계획 수립 | AI Assistant |
| 2025-01-20 | 1.1.0 | 베이스 래퍼 구현 완료 | AI Assistant |

---

*이 문서는 AI Data Science Team A2A 래퍼 마이그레이션의 전체 로드맵을 제공합니다. 각 단계별로 체크리스트를 완료하며 진행하시기 바랍니다.* 