# 🚀 CherryAI A2A 통합 시스템 설계 계획

## 🎯 프로젝트 목표

**현재 3개 분산 시스템 → 1개 통합 A2A 시스템**

### 현재 문제점
- ❌ **분산 시스템**: A2A(9개 서버) + Standalone(1개) + 독립 Agent(1개) = 11개 서버
- ❌ **데이터 불일치**: 서버별 별도 데이터 저장소
- ❌ **복잡한 관리**: 11개 포트, 11개 프로세스
- ❌ **호환성 문제**: A2A SDK 버전별 구현 차이

### 통합 목표
- ✅ **단일 A2A 서버**: 모든 기능을 하나의 포트(8100)에서 제공
- ✅ **A2A SDK 0.2.9 완전 준수**: 표준 패턴 100% 적용
- ✅ **통합 데이터 관리**: 단일 데이터 저장소
- ✅ **검증된 기능 보존**: Standalone 서버의 안정성 유지

## 🏗️ 통합 아키텍처 설계

### Phase 1: Universal A2A Agent 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                Universal A2A Agent (Port 8100)             │
├─────────────────────────────────────────────────────────────┤
│  A2A SDK 0.2.9 Standard Layer                             │
│  ├── AgentExecutor (execute/cancel)                       │
│  ├── RequestContext (message parsing)                     │
│  ├── TaskUpdater (streaming responses)                    │
│  └── AgentCard (skills & capabilities)                    │
├─────────────────────────────────────────────────────────────┤
│  Skill-Based Orchestration Layer                          │
│  ├── DataAnalysisSkill (자연어 분석)                      │
│  ├── DataVisualizationSkill (시각화)                     │
│  ├── DataCleaningSkill (정리)                             │
│  ├── StatisticsSkill (통계)                               │
│  └── MLSkill (머신러닝)                                   │
├─────────────────────────────────────────────────────────────┤
│  Unified Core Components (Standalone에서 검증됨)          │
│  ├── PandasAgentCore (자연어 처리)                        │
│  ├── MultiDataFrameHandler (데이터 관리)                  │
│  ├── NaturalLanguageProcessor (한국어/영어)               │
│  └── IntelligentDataHandler (파일 처리)                   │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├── UnifiedDataManager (단일 데이터 스토어)               │
│  ├── SessionManager (세션 관리)                           │
│  └── FileManager (파일 업로드/다운로드)                    │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Agent Skills 정의

기존 9개 분산 에이전트 → 5개 통합 스킬

```python
UNIFIED_SKILLS = [
    AgentSkill(
        id="natural_language_analysis",
        name="자연어 데이터 분석",
        description="자연어로 데이터를 질문하면 pandas 코드를 생성하고 실행하여 답변",
        tags=["nlp", "pandas", "analysis", "korean"]
    ),
    AgentSkill(
        id="data_visualization", 
        name="데이터 시각화",
        description="차트, 그래프, 히트맵 등 다양한 시각화 자동 생성",
        tags=["visualization", "matplotlib", "seaborn", "plotly"]
    ),
    AgentSkill(
        id="data_quality_management",
        name="데이터 품질 관리", 
        description="데이터 정리, 결측치 처리, 이상치 탐지, 중복 제거",
        tags=["cleaning", "quality", "preprocessing"]
    ),
    AgentSkill(
        id="statistical_analysis",
        name="통계 분석",
        description="기술통계, 추론통계, 상관분석, 분포 분석",
        tags=["statistics", "correlation", "distribution"]
    ),
    AgentSkill(
        id="machine_learning",
        name="머신러닝",
        description="모델 훈련, 평가, 예측, 하이퍼파라미터 튜닝",
        tags=["ml", "sklearn", "prediction", "model"]
    )
]
```

## 🛠️ 구현 전략

### Step 1: 핵심 컴포넌트 통합 (1주)

#### 1.1 Universal A2A Agent 생성
```python
class UniversalDataScienceAgent(AgentExecutor):
    """통합 데이터 사이언스 A2A 에이전트"""
    
    def __init__(self):
        # 검증된 컴포넌트들 통합
        self.pandas_core = PandasAgentCore()
        self.data_handler = MultiDataFrameHandler() 
        self.nlp_processor = NaturalLanguageProcessor()
        self.visualizer = DataVisualizer()
        self.ml_engine = MLEngine()
        
        # A2A 표준 컴포넌트
        self.skill_router = SkillBasedRouter()
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        """A2A SDK 0.2.9 표준 execute 메서드"""
        
        # 1. 사용자 입력 파싱
        user_input = self._extract_user_input(context)
        
        # 2. 스킬 라우팅
        target_skill = await self.skill_router.route(user_input)
        
        # 3. 적절한 핸들러로 전달
        response = await self._execute_skill(target_skill, user_input, task_updater)
        
        # 4. A2A 표준 응답
        await task_updater.update_status(
            TaskState.completed,
            message=response,
            final=True
        )
```

#### 1.2 스킬 기반 라우팅 시스템
```python
class SkillBasedRouter:
    """사용자 요청을 적절한 스킬로 라우팅"""
    
    SKILL_PATTERNS = {
        "natural_language_analysis": [
            "분석", "요약", "설명", "알려줘", "보여줘", "어떻게"
        ],
        "data_visualization": [
            "차트", "그래프", "시각화", "플롯", "히트맵", "분포도"
        ],
        "data_quality_management": [
            "정리", "결측", "이상치", "중복", "전처리", "정제"
        ],
        "statistical_analysis": [
            "통계", "상관", "분포", "평균", "표준편차", "검정"
        ],
        "machine_learning": [
            "모델", "예측", "훈련", "분류", "회귀", "군집"
        ]
    }
    
    async def route(self, user_input: str) -> str:
        """사용자 입력을 분석하여 적절한 스킬 반환"""
        # LLM 기반 인텐트 분류 또는 키워드 매칭
        return self._classify_intent(user_input)
```

### Step 2: 데이터 레이어 통합 (3일)

#### 2.1 통합 데이터 매니저
```python
class UnifiedDataManager:
    """단일 데이터 저장소 관리"""
    
    def __init__(self):
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.sessions: Dict[str, SessionData] = {}
        
    async def upload_data(self, data, session_id: str = None) -> str:
        """데이터 업로드 (Excel, CSV, JSON 지원)"""
        
    async def get_dataframes(self, session_id: str = None) -> List[pd.DataFrame]:
        """세션별 데이터프레임 조회"""
        
    async def process_natural_query(self, query: str, session_id: str = None):
        """자연어 쿼리 처리"""
```

#### 2.2 A2A 표준 파일 업로드
```python
# A2A SDK 0.2.9의 파일 처리 표준 준수
async def handle_file_upload(self, context: RequestContext) -> str:
    """A2A 표준 방식으로 파일 업로드 처리"""
    
    # A2A 메시지에서 파일 추출
    for part in context.message.parts:
        if hasattr(part, 'file'):
            file_data = part.file
            # 파일 처리 로직
            
    return dataframe_id
```

### Step 3: A2A 표준 준수 (2일)

#### 3.1 표준 Agent Card
```python
def create_universal_agent_card() -> AgentCard:
    return AgentCard(
        name="Universal Data Science Agent",
        description="통합 데이터 사이언스 에이전트 - 자연어 분석부터 머신러닝까지",
        version="1.0.0",
        url="http://localhost:8100",
        defaultInputModes=["text", "application/json"],
        defaultOutputModes=["text", "application/json", "image/png"],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True,
            supports_files=True
        ),
        skills=UNIFIED_SKILLS
    )
```

#### 3.2 스트리밍 응답 구현
```python
async def stream_response(self, query: str, task_updater: TaskUpdater):
    """실시간 스트리밍 응답"""
    
    # 작업 시작 알림
    await task_updater.update_status(
        TaskState.working,
        message="🔍 데이터를 분석하고 있습니다..."
    )
    
    # 중간 진행 상황
    await task_updater.update_status(
        TaskState.working, 
        message="📊 통계를 계산하고 있습니다..."
    )
    
    # 최종 결과
    await task_updater.update_status(
        TaskState.completed,
        message=final_result,
        final=True
    )
```

## 📋 마이그레이션 계획

### Week 1: 기반 구축
- [x] 현재 시스템 분석 완료
- [ ] UniversalDataScienceAgent 기본 구조 구현
- [ ] 핵심 컴포넌트 통합 (PandasCore, DataHandler, NLP)
- [ ] 기본 A2A 서버 설정

### Week 2: 스킬 시스템 구현  
- [ ] SkillBasedRouter 구현
- [ ] 5개 핵심 스킬 구현
- [ ] 자연어 쿼리 처리 통합
- [ ] 파일 업로드/다운로드 A2A 표준화

### Week 3: 고급 기능 및 최적화
- [ ] 실시간 스트리밍 구현
- [ ] 세션 관리 시스템
- [ ] 에러 처리 및 복구
- [ ] 성능 최적화

### Week 4: 테스트 및 배포
- [ ] 통합 테스트 (pytest + Playwright MCP)
- [ ] 레거시 시스템과 호환성 테스트
- [ ] 문서화 및 API 가이드
- [ ] 프로덕션 배포

## 🧪 테스트 전략

### 1. A2A SDK 0.2.9 표준 준수 테스트
```python
def test_a2a_standard_compliance():
    """A2A SDK 0.2.9 표준 준수 검증"""
    
    # Agent Card 표준 검증
    assert agent_card.name
    assert agent_card.description
    assert agent_card.skills
    
    # AgentExecutor 표준 검증  
    assert hasattr(executor, 'execute')
    assert hasattr(executor, 'cancel')
    
    # 메시지 형식 검증
    # ...
```

### 2. 기능 호환성 테스트
```python  
def test_standalone_feature_parity():
    """Standalone 서버 기능과 동등성 검증"""
    
    # 자연어 쿼리 처리
    response1 = standalone_server.query("데이터 요약")
    response2 = a2a_server.query("데이터 요약")
    assert response1.content_similarity(response2) > 0.9
    
    # 파일 업로드
    # 시각화 생성
    # ...
```

### 3. 성능 테스트
```python
def test_performance_benchmarks():
    """통합 시스템 성능 검증"""
    
    # 응답 시간 < 3초
    # 메모리 사용량 < 1GB  
    # 동시 사용자 > 10명
```

## 🎯 성공 지표

### 기술적 지표
- ✅ A2A SDK 0.2.9 표준 100% 준수
- ✅ 기존 기능 100% 보존 (Standalone 대비)
- ✅ 응답 시간 < 3초 (90% 쿼리)
- ✅ 동시 사용자 지원 > 10명

### 운영적 지표  
- ✅ 서버 수: 11개 → 1개 (91% 감소)
- ✅ 포트 수: 11개 → 1개 (91% 감소)
- ✅ 메모리 사용량 < 1GB
- ✅ 가용성 > 99%

### 사용자 경험 지표
- ✅ API 단순화: 1개 엔드포인트로 모든 기능
- ✅ 설정 간소화: 1개 설정 파일
- ✅ 문서화: 단일 API 가이드
- ✅ 학습 비용: 기존 대비 50% 감소

## 🚀 실행 명령

### 통합 시스템 시작
```bash
# 기존 (11개 서버)
./ai_ds_team_system_start.sh

# 통합 후 (1개 서버)
python unified_a2a_agent.py --port 8100
```

### API 사용
```bash
# 기존 (여러 엔드포인트)
curl -X POST http://localhost:8080/api/upload ...
curl -X POST http://localhost:8307/ai_ds_team/data_loader ...
curl -X POST http://localhost:8312/ai_ds_team/eda_tools ...

# 통합 후 (단일 A2A 엔드포인트)
curl -X POST http://localhost:8100/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"message": {"parts": [{"text": "데이터를 업로드하고 분석해주세요"}]}}'
```

## 📈 예상 효과

### 1. 복잡성 대폭 감소
- **현재**: 11개 서버, 11개 포트, 11개 프로세스, 3개 다른 프로토콜
- **통합 후**: 1개 서버, 1개 포트, 1개 프로세스, 1개 A2A 프로토콜

### 2. 성능 향상
- **메모리**: 11개 프로세스 → 1개 프로세스 (메모리 사용량 70% 감소)
- **통신**: 서버 간 HTTP 호출 제거 → 내부 함수 호출
- **데이터**: 중복 로딩 제거 → 단일 메모리 공간

### 3. 개발 생산성 향상
- **코드 중복 제거**: 11개 서버의 공통 로직 통합
- **테스트 간소화**: 1개 서버만 테스트하면 됨
- **배포 간소화**: 1개 컨테이너로 전체 시스템 배포

### 4. 사용자 경험 향상
- **학습 비용**: 1개 API만 학습하면 모든 기능 사용
- **설정 복잡성**: 1개 설정 파일로 전체 시스템 구성
- **디버깅**: 1개 로그 파일에서 모든 문제 추적

이 통합 계획을 통해 CherryAI는 **복잡한 분산 시스템에서 단순하고 강력한 통합 시스템**으로 진화할 것입니다. 🚀 