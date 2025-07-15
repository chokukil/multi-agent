# 🍒 CherryAI 통합 데이터 로딩 시스템 설계 문서

**프로젝트**: CherryAI LLM First Architecture  
**문서 버전**: 1.0  
**작성일**: 2025년 1월 27일  
**기준**: pandas_agent 패턴을 활용한 12개 A2A 에이전트 표준화  

---

## 📊 1. 개요 및 목적

### 1.1 배경
현재 CherryAI 시스템의 12개 A2A 에이전트들이 각각 다른 데이터 로딩 방식을 사용하여 다음과 같은 문제들이 발생하고 있습니다:

- **일관성 부족**: 6가지 서로 다른 데이터 로딩 패턴 사용
- **신뢰성 문제**: UTF-8 인코딩 오류, 파일 선택 실패
- **성능 저하**: 캐싱 없음, 중복 로딩
- **유지보수 어려움**: 에이전트별 개별 로직 관리

### 1.2 목적
pandas_agent의 우수한 아키텍처를 기준으로 모든 A2A 에이전트의 데이터 로딩을 표준화하여:

- **LLM First 원칙** 완전 준수
- **A2A SDK 0.2.9 표준** 완벽 적용  
- **100% 기능 유지** 하면서 데이터 계층만 통합
- **Mock 사용 금지**, 실제 동작하는 통합 시스템 구축

---

## 🔍 2. 현재 상황 분석

### 2.1 12개 A2A 에이전트 완전 현황 분석

| # | Agent | Port | 현재 데이터 로딩 방식 | 주요 문제점 | 우선순위 | 마이그레이션 복잡도 |
|---|-------|------|---------------------|------------|---------|-------------------|
| 1 | **orchestrator** | 8100 | Agent Registry 관리만 | 데이터 직접 처리 안함 | LOW | 🟢 단순 |
| 2 | **data_cleaning** | 8306 | IntelligentDataHandler | 빈 데이터 오류, 컬럼 없음 에러 | HIGH | 🟡 중간 |
| 3 | **data_loader** | 8307 | AIDataScienceTeamWrapper | 다중 구현체 혼재, 불일치 | CRITICAL | 🔴 복잡 |
| 4 | **data_visualization** | 8308 | SafeDataLoader | UTF-8 인코딩 오류, 차트 생성 실패 | HIGH | 🟡 중간 |
| 5 | **data_wrangling** | 8309 | UnifiedDataLoader | 파일 선택 불안정, 변환 오류 | HIGH | 🟡 중간 |
| 6 | **feature_engineering** | 8310 | SafeDataLoader 직접 호출 | 캐싱 없음, 반복 로딩 | MEDIUM | 🟢 단순 |
| 7 | **sql_database** | 8311 | 직접 pandas 로딩 | 예외 처리 부족, DB 연결 불안정 | MEDIUM | 🟡 중간 |
| 8 | **eda_tools** | 8312 | 파일 스캔 + 직접 로딩 | 인코딩 문제, 통계 계산 오류 | HIGH | 🟡 중간 |
| 9 | **h2o_ml** | 8313 | SafeDataLoader | 에러 처리 부족, 모델링 실패 | MEDIUM | 🟡 중간 |
| 10 | **mlflow_tools** | 8314 | SafeDataLoader | 동일한 패턴, 실험 추적 불안정 | MEDIUM | 🟢 단순 |
| 11 | **pandas_agent** | 8210 | **FileConnector 패턴** ⭐ | **기준 모델 - 우수함** | **TEMPLATE** | ✅ **기준** |
| 12 | **report_generator** | 8316 | Agent 결과 수집 | 다른 에이전트 의존, 종합 분석 한계 | LOW | 🟢 단순 |

**📊 전체 통계:**
- **CRITICAL**: 1개 (data_loader)
- **HIGH**: 4개 (data_cleaning, data_visualization, data_wrangling, eda_tools)  
- **MEDIUM**: 4개 (feature_engineering, sql_database, h2o_ml, mlflow_tools)
- **LOW**: 2개 (orchestrator, report_generator)
- **TEMPLATE**: 1개 (pandas_agent - 기준 모델)

### 2.2 pandas_agent 우수 패턴 분석

```python
# 🎯 pandas_agent의 핵심 아키텍처
class PandasAgent:
    """LLM First + FileConnector Pattern의 완벽한 구현"""
    
    # 1. LLM 통합 데이터 처리
    def load_dataframe(self, df: pd.DataFrame, name: str = "main") -> str
    def load_from_file(self, file_path: str, name: str = "main") -> str
    
    # 2. FileConnector 확장성
    async def _handle_data_loading(self, query_analysis, task_updater)
    
    # 3. SmartDataFrame 지능형 처리
    # 4. Cache Manager 성능 최적화
    # 5. A2A TaskUpdater 완벽 통합
```

**⭐ 핵심 우수성:**
- **5단계 LLM 처리 파이프라인**: 의도분석 → 코드생성 → 실행 → 결과해석 → 시각화
- **Connector Pattern**: 확장 가능한 데이터 소스 지원
- **지능형 캐싱**: LRU + TTL + 태그 기반 최적화
- **완벽한 A2A 통합**: TaskUpdater + 실시간 스트리밍

---

## 🏗️ 3. 통합 시스템 아키텍처

### 3.1 전체 아키텍처 설계

```
📊 CherryAI Unified Data Loading System Architecture

┌─────────────────────────────────────────────────────────────────┐
│                    🎯 A2A Agent Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  [data_cleaning] [data_visualization] [eda_tools] ... [12개]    │
│                          ↓                                     │
│              📝 UnifiedDataInterface                           │
│                    (표준화된 API)                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                🧠 LLM First Data Engine                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │🔍 Intent    │  │🎯 File      │  │📊 Smart     │              │
│  │  Analyzer   │  │  Selector   │  │  DataFrame  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                📁 Unified Connector Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │📄 File      │  │🗄️ SQL       │  │🌐 API       │              │
│  │  Connector  │  │  Connector  │  │  Connector  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                🚀 Performance & Caching Layer                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │💾 Cache     │  │⚡ Async     │  │🔒 Security  │              │
│  │  Manager    │  │  Pipeline   │  │  Manager    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 핵심 컴포넌트 설계

#### 3.2.1 UnifiedDataInterface (표준 API)

```python
class UnifiedDataInterface:
    """모든 A2A 에이전트가 사용할 표준 데이터 인터페이스"""
    
    # 필수 메서드 (모든 에이전트 구현 필수)
    async def load_data(self, intent: DataIntent, context: A2AContext) -> SmartDataFrame
    async def get_data_info(self) -> DataProfile
    async def validate_data_quality(self) -> QualityReport
    
    # 선택적 메서드 (에이전트별 특화)
    async def transform_data(self, operations: List[Operation]) -> SmartDataFrame
    async def cache_data(self, key: str, ttl: int = 3600) -> bool
```

#### 3.2.2 LLMFirstDataEngine (핵심 엔진)

```python
class LLMFirstDataEngine:
    """LLM 기반 지능형 데이터 처리 엔진"""
    
    async def analyze_intent(self, user_query: str, context: A2AContext) -> DataIntent
    async def select_optimal_file(self, intent: DataIntent, available_files: List[str]) -> str
    async def create_smart_dataframe(self, df: pd.DataFrame, metadata: Dict) -> SmartDataFrame
    async def optimize_loading_strategy(self, file_info: FileInfo) -> LoadConfig
```

---

## 📋 4. 에이전트별 마이그레이션 계획

### 4.1 우선순위 그룹 분류 (12개 에이전트 완전 커버리지)

#### 🔴 CRITICAL (즉시 처리 필요 - 1개)
- **data_loader (8307)**: 모든 에이전트의 데이터 공급원, 다중 구현체 혼재 문제 해결 시급

#### 🟡 HIGH (1주 내 처리 필요 - 4개)
- **data_cleaning (8306)**: 빈 데이터 오류, "Cannot describe DataFrame without columns" 해결 시급
- **data_visualization (8308)**: UTF-8 인코딩 문제, 차트 생성 실패 해결
- **data_wrangling (8309)**: 파일 선택 불안정성, 데이터 변환 오류 해결
- **eda_tools (8312)**: 인코딩 문제, 통계 계산 오류 해결 (주요 분석 도구)

#### 🟢 MEDIUM (2주 내 처리 - 4개)
- **feature_engineering (8310)**: 캐싱 시스템 도입, 반복 로딩 최적화
- **sql_database (8311)**: 예외 처리 강화, DB 연결 안정성 개선
- **h2o_ml (8313)**: 에러 처리 강화, 모델링 안정성 개선
- **mlflow_tools (8314)**: 표준화 적용, 실험 추적 안정성 개선

#### 🔵 LOW (마지막 처리 - 2개)
- **orchestrator (8100)**: 데이터 직접 처리 안함, Agent Registry 관리만
- **report_generator (8316)**: 다른 에이전트 결과 수집 및 종합, 의존성 관리

#### ⭐ TEMPLATE (기준 모델 - 1개)
- **pandas_agent (8210)**: FileConnector 패턴 기준 모델, 모든 에이전트가 따라야 할 표준

### 4.2 상세 마이그레이션 가이드

#### 4.2.1 data_loader (8307) - CRITICAL

**현재 문제점:**
- 3가지 다른 구현체 혼재 (AIDataScienceTeamWrapper, DataLoaderAgent, 직접 pandas)
- A2A 표준 불일치
- 에러 처리 부족

**마이그레이션 계획:**
```python
# 1단계: 기존 코드 분석 및 백업
# 파일: a2a_ds_servers/ai_ds_team_data_loader_server.py
# 백업: a2a_ds_servers/backup/data_loader_original.py

# 2단계: UnifiedDataInterface 구현
class DataLoaderExecutor(AgentExecutor, UnifiedDataInterface):
    def __init__(self):
        self.data_engine = LLMFirstDataEngine()
        self.file_connector = FileConnector()
        self.cache_manager = CacheManager()
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # pandas_agent 패턴 적용
        user_query = self._extract_user_query(context)
        intent = await self.data_engine.analyze_intent(user_query, context)
        smart_df = await self.load_data(intent, context)
        
        # A2A 표준 응답
        await self._send_a2a_response(smart_df, task_updater)

# 3단계: A2A SDK 0.2.9 표준 적용
# - TaskUpdater 패턴 완벽 구현
# - TextPart + JSON 직렬화
# - 실시간 스트리밍 지원

# 4단계: 테스트 및 검증
# - 기존 기능 100% 보장
# - 성능 개선 확인
# - 에러 처리 강화
```

**구체적 구현 지침:**
1. **기존 기능 보존**: `DataLoaderToolsAgent` 완전 호환
2. **에러 처리**: UTF-8, 파일 없음, 권한 문제 대응
3. **성능 최적화**: 파일 크기별 로딩 전략 수립
4. **A2A 통합**: TaskUpdater + 아티팩트 생성

#### 4.2.2 data_cleaning (8306) - HIGH

**현재 문제점:**
- `IntelligentDataHandler` 사용하지만 빈 데이터 오류
- `Cannot describe a DataFrame without columns` 에러

**마이그레이션 계획:**
```python
# 문제 해결 전략
class DataCleaningExecutor(AgentExecutor, UnifiedDataInterface):
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. 데이터 검증 강화
        smart_df = await self.load_data(intent, context)
        
        if smart_df.is_empty():
            await self._handle_empty_data(context, task_updater)
            return
        
        # 2. 데이터 정제 실행
        cleaning_results = await self._perform_cleaning(smart_df, intent)
        
        # 3. 결과 검증 및 반환
        await self._validate_and_return(cleaning_results, task_updater)

    async def _handle_empty_data(self, context, task_updater):
        """빈 데이터 전용 처리 로직"""
        await task_updater.update_status(
            TaskState.completed,
            message="⚠️ 데이터가 비어있습니다. 다른 파일을 시도하거나 데이터를 업로드해주세요."
        )
```

**구체적 구현 지침:**
1. **빈 데이터 감지**: 로딩 즉시 검증
2. **사용자 안내**: 명확한 에러 메시지와 해결책 제공
3. **폴백 전략**: 다른 파일 자동 시도
4. **품질 보고서**: 정제 전후 비교 리포트

#### 4.2.3 data_visualization (8308) - HIGH

**현재 문제점:**
- UTF-8 인코딩 오류
- SafeDataLoader 사용하지만 제한적

**마이그레이션 계획:**
```python
class DataVisualizationExecutor(AgentExecutor, UnifiedDataInterface):
    def __init__(self):
        self.visualization_engine = VisualizationEngine()
        self.encoding_handler = EncodingHandler()
    
    async def load_data(self, intent: DataIntent, context: A2AContext) -> SmartDataFrame:
        # pandas_agent 패턴 + 인코딩 처리 강화
        file_path = await self.data_engine.select_optimal_file(intent, available_files)
        
        # 다중 인코딩 시도
        for encoding in ['utf-8', 'cp949', 'euc-kr', 'latin1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        return SmartDataFrame(df, metadata={'encoding': encoding})
```

**구체적 구현 지침:**
1. **인코딩 자동 감지**: 다중 인코딩 시도 로직
2. **시각화 최적화**: 데이터 타입별 최적 차트 추천
3. **interactive 차트**: Plotly + Streamlit 통합
4. **메모리 최적화**: 대용량 데이터 샘플링

#### 4.2.4 eda_tools (8312) - HIGH

**현재 문제점:**
- 파일 스캔 후 직접 로딩
- 인코딩 문제 존재

**마이그레이션 계획:**
```python
class EDAToolsExecutor(AgentExecutor, UnifiedDataInterface):
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # pandas_agent의 5단계 분석 파이프라인 적용
        
        # 1. 의도 분석
        eda_intent = await self._analyze_eda_intent(user_query)
        
        # 2. 데이터 로딩 (통합 인터페이스)
        smart_df = await self.load_data(eda_intent, context)
        
        # 3. EDA 분석 수행
        eda_results = await self._perform_comprehensive_eda(smart_df, eda_intent)
        
        # 4. 인사이트 생성 (LLM)
        insights = await self._generate_insights(eda_results, smart_df)
        
        # 5. 결과 포맷팅 및 반환
        await self._format_and_return_results(eda_results, insights, task_updater)
```

**구체적 구현 지침:**
1. **포괄적 EDA**: 기술통계, 분포, 상관관계, 이상값
2. **LLM 인사이트**: 패턴 발견 및 해석
3. **시각화 통합**: 차트 + 표 + 텍스트 조합
4. **성능 최적화**: 대용량 데이터 처리

#### 4.2.5 data_wrangling (8309) - HIGH

**현재 문제점:**
- UnifiedDataLoader 사용하지만 파일 선택 불안정
- 데이터 변환 과정에서 오류 발생

**마이그레이션 계획:**
```python
class DataWranglingExecutor(AgentExecutor, UnifiedDataInterface):
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # pandas_agent 패턴 + 변환 특화
        
        # 1. 변환 의도 분석 (LLM)
        wrangling_intent = await self._analyze_wrangling_intent(user_query)
        
        # 2. 안전한 데이터 로딩
        smart_df = await self.load_data(wrangling_intent, context)
        
        # 3. 변환 작업 실행
        transformed_df = await self._perform_transformations(smart_df, wrangling_intent)
        
        # 4. 변환 결과 검증
        await self._validate_transformations(transformed_df, task_updater)
```

**구체적 구현 지침:**
1. **변환 안전성**: 원본 데이터 백업 후 변환 수행
2. **LLM 가이드**: 변환 로직을 LLM이 동적 생성
3. **단계별 검증**: 각 변환 단계마다 데이터 무결성 확인
4. **롤백 기능**: 변환 실패 시 이전 상태로 복구

### 4.3 MEDIUM 우선순위 에이전트 마이그레이션 (4개)

#### 4.3.1 feature_engineering (8310) - MEDIUM

**현재 문제점:**
- SafeDataLoader 직접 호출, 캐싱 없음
- 반복적인 파일 로딩으로 성능 저하

**마이그레이션 계획:**
```python
class FeatureEngineeringExecutor(AgentExecutor, UnifiedDataInterface):
    def __init__(self):
        super().__init__()
        self.feature_cache = FeatureCache()  # 특성 캐싱 시스템
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. 특성 엔지니어링 의도 분석
        feature_intent = await self._analyze_feature_intent(user_query)
        
        # 2. 캐시된 특성 확인
        cached_features = await self.feature_cache.get(feature_intent.cache_key)
        
        # 3. 데이터 로딩 (캐시 미스 시만)
        if not cached_features:
            smart_df = await self.load_data(feature_intent, context)
            features = await self._engineer_features(smart_df, feature_intent)
            await self.feature_cache.set(feature_intent.cache_key, features)
        
        # 4. 특성 선택 및 최적화
        optimized_features = await self._optimize_features(features, feature_intent)
```

#### 4.3.2 sql_database (8311) - MEDIUM

**현재 문제점:**
- 직접 pandas 로딩, 예외 처리 부족
- DB 연결 불안정성

**마이그레이션 계획:**
```python
class SQLDatabaseExecutor(AgentExecutor, UnifiedDataInterface):
    def __init__(self):
        super().__init__()
        self.sql_connector = SQLConnector()  # pandas_agent 패턴
        self.connection_pool = ConnectionPool()
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. SQL 의도 분석
        sql_intent = await self._analyze_sql_intent(user_query)
        
        # 2. 연결 풀에서 안전한 DB 연결
        async with self.connection_pool.get_connection() as conn:
            # 3. SQL 쿼리 생성 (LLM)
            sql_query = await self._generate_sql_query(sql_intent)
            
            # 4. 쿼리 실행 및 DataFrame 변환
            smart_df = await self.sql_connector.execute_query(sql_query, conn)
            
            # 5. 결과 분석 및 인사이트 생성
            insights = await self._analyze_sql_results(smart_df, sql_intent)
```

#### 4.3.3 h2o_ml (8313) - MEDIUM

**현재 문제점:**
- SafeDataLoader 사용, 에러 처리 부족
- 모델링 과정에서 안정성 문제

**마이그레이션 계획:**
```python
class H2OMLExecutor(AgentExecutor, UnifiedDataInterface):
    def __init__(self):
        super().__init__()
        self.h2o_manager = H2OManager()
        self.model_cache = ModelCache()
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. ML 의도 분석
        ml_intent = await self._analyze_ml_intent(user_query)
        
        # 2. 데이터 로딩 및 ML 준비
        smart_df = await self.load_data(ml_intent, context)
        h2o_frame = await self._convert_to_h2o_frame(smart_df)
        
        # 3. H2O 환경 초기화
        await self.h2o_manager.ensure_cluster()
        
        # 4. AutoML 실행
        automl_results = await self._run_h2o_automl(h2o_frame, ml_intent)
        
        # 5. 모델 해석 및 결과 분석
        interpretations = await self._interpret_models(automl_results)
```

#### 4.3.4 mlflow_tools (8314) - MEDIUM

**현재 문제점:**
- SafeDataLoader 동일 패턴
- 실험 추적 불안정성

**마이그레이션 계획:**
```python
class MLflowToolsExecutor(AgentExecutor, UnifiedDataInterface):
    def __init__(self):
        super().__init__()
        self.mlflow_client = MLflowClient()
        self.experiment_tracker = ExperimentTracker()
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. 실험 의도 분석
        experiment_intent = await self._analyze_experiment_intent(user_query)
        
        # 2. 데이터 로딩 및 실험 설정
        smart_df = await self.load_data(experiment_intent, context)
        
        # 3. MLflow 실험 시작
        with self.mlflow_client.start_run() as run:
            # 4. 모델 학습 및 추적
            model_results = await self._train_and_track_model(smart_df, experiment_intent)
            
            # 5. 실험 결과 로깅
            await self._log_experiment_results(model_results, run)
```

### 4.4 LOW 우선순위 에이전트 마이그레이션 (2개)

#### 4.4.1 orchestrator (8100) - LOW

**현재 문제점:**
- 데이터 직접 처리 안함, Agent Registry 관리만

**마이그레이션 계획:**
```python
class OrchestratorExecutor(AgentExecutor):
    # 데이터 직접 처리 없음, Agent 발견 및 조정만
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. 사용자 요청 분석
        orchestration_plan = await self._analyze_orchestration_needs(user_query)
        
        # 2. 에이전트 발견 및 계획 수립
        available_agents = await self._discover_agents()
        execution_plan = await self._create_execution_plan(orchestration_plan, available_agents)
        
        # 3. 에이전트 실행 조정 (데이터는 각 에이전트가 개별 로딩)
        results = await self._execute_multi_agent_plan(execution_plan)
```

#### 4.4.2 report_generator (8316) - LOW

**현재 문제점:**
- 다른 에이전트 결과 수집, 종합 분석 한계

**마이그레이션 계획:**
```python
class ReportGeneratorExecutor(AgentExecutor, UnifiedDataInterface):
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. 보고서 요구사항 분석
        report_intent = await self._analyze_report_intent(user_query)
        
        # 2. 데이터 소스 식별 (원본 데이터 + 에이전트 결과)
        data_sources = await self._identify_data_sources(report_intent)
        
        # 3. 통합 데이터 로딩
        unified_data = await self._load_unified_data(data_sources, context)
        
        # 4. 종합 보고서 생성
        comprehensive_report = await self._generate_comprehensive_report(unified_data, report_intent)
```

### 4.5 공통 마이그레이션 패턴

#### 모든 에이전트 공통 템플릿:
```python
# 표준화된 마이그레이션 템플릿 (12개 에이전트 공통)
class StandardUnifiedAgentExecutor(AgentExecutor, UnifiedDataInterface):
    def __init__(self, agent_name: str, specialized_config: Dict[str, Any]):
        super().__init__()
        self.agent_name = agent_name
        self.data_engine = LLMFirstDataEngine()
        self.cache_manager = CacheManager()
        self.specialized_processor = self._create_specialized_processor(specialized_config)
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater):
        # 1. 표준 데이터 로딩 (pandas_agent 패턴)
        intent = await self.data_engine.analyze_intent(user_query, context)
        smart_df = await self.load_data(intent, context)
        
        # 2. 에이전트별 특화 처리
        results = await self.specialized_processor.process(smart_df, intent)
        
        # 3. 표준 A2A 응답
        await self._send_standard_response(results, task_updater)
```

---

## 🔧 5. 구현 세부사항

### 5.1 핵심 클래스 설계

#### 5.1.1 UnifiedDataInterface

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class DataIntentType(Enum):
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    MODELING = "modeling"

@dataclass
class DataIntent:
    intent_type: DataIntentType
    confidence: float
    file_preferences: List[str]
    operations: List[str]
    constraints: Dict[str, Any]

@dataclass
class DataProfile:
    shape: tuple
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    memory_usage: int
    encoding: str
    file_size: int

@dataclass
class QualityReport:
    overall_score: float
    completeness: float
    consistency: float
    validity: float
    issues: List[str]
    recommendations: List[str]

class UnifiedDataInterface(ABC):
    """모든 A2A 에이전트가 구현해야 할 표준 데이터 인터페이스"""
    
    @abstractmethod
    async def load_data(self, intent: DataIntent, context: 'A2AContext') -> 'SmartDataFrame':
        """데이터 로딩 (필수 구현)"""
        pass
    
    @abstractmethod
    async def get_data_info(self) -> DataProfile:
        """데이터 정보 조회 (필수 구현)"""
        pass
    
    @abstractmethod
    async def validate_data_quality(self) -> QualityReport:
        """데이터 품질 검증 (필수 구현)"""
        pass
    
    # 선택적 구현 메서드들
    async def transform_data(self, operations: List['Operation']) -> 'SmartDataFrame':
        """데이터 변환 (선택적 구현)"""
        raise NotImplementedError("This agent doesn't support data transformation")
    
    async def cache_data(self, key: str, ttl: int = 3600) -> bool:
        """데이터 캐싱 (선택적 구현)"""
        return False  # 기본적으로 캐싱 비활성화
```

#### 5.1.2 LLMFirstDataEngine

```python
class LLMFirstDataEngine:
    """LLM 기반 지능형 데이터 처리 엔진"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client or self._create_llm_client()
        self.intent_analyzer = IntentAnalyzer(self.llm)
        self.file_selector = IntelligentFileSelector(self.llm)
        self.encoding_detector = EncodingDetector()
    
    async def analyze_intent(self, user_query: str, context: 'A2AContext') -> DataIntent:
        """사용자 의도 분석"""
        
        # LLM 프롬프트 구성
        prompt = f"""
        사용자 요청을 분석하여 데이터 처리 의도를 파악해주세요:
        
        요청: {user_query}
        컨텍스트: {context.to_dict()}
        
        다음 형식으로 응답해주세요:
        {{
            "intent_type": "analysis|visualization|cleaning|transformation|modeling",
            "confidence": 0.0-1.0,
            "file_preferences": ["특정 파일명이나 패턴"],
            "operations": ["수행할 작업들"],
            "constraints": {{"제약조건들": "값"}}
        }}
        """
        
        response = await self.llm.agenerate([prompt])
        intent_data = json.loads(response.generations[0][0].text)
        
        return DataIntent(
            intent_type=DataIntentType(intent_data['intent_type']),
            confidence=intent_data['confidence'],
            file_preferences=intent_data['file_preferences'],
            operations=intent_data['operations'],
            constraints=intent_data['constraints']
        )
    
    async def select_optimal_file(self, intent: DataIntent, available_files: List[str]) -> str:
        """최적 파일 선택"""
        
        if not available_files:
            raise ValueError("사용 가능한 파일이 없습니다")
        
        # 파일 정보 수집
        file_infos = []
        for file_path in available_files:
            info = await self._analyze_file_info(file_path)
            file_infos.append(info)
        
        # LLM 기반 파일 선택
        prompt = f"""
        사용자 의도에 가장 적합한 파일을 선택해주세요:
        
        의도: {intent.intent_type.value}
        선호도: {intent.file_preferences}
        작업: {intent.operations}
        
        사용 가능한 파일들:
        {json.dumps(file_infos, indent=2, ensure_ascii=False)}
        
        가장 적합한 파일의 경로만 반환해주세요.
        """
        
        response = await self.llm.agenerate([prompt])
        selected_file = response.generations[0][0].text.strip()
        
        return selected_file
```

#### 5.1.3 SmartDataFrame

```python
class SmartDataFrame:
    """지능형 DataFrame 클래스 (pandas_agent 패턴)"""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict[str, Any] = None):
        self.df = df
        self.metadata = metadata or {}
        self.profile: Optional[DataProfile] = None
        self.quality_report: Optional[QualityReport] = None
        self._cache_info = {}
    
    @property
    def shape(self) -> tuple:
        return self.df.shape
    
    def is_empty(self) -> bool:
        """빈 데이터 검사"""
        return self.df.empty or self.df.shape[0] == 0 or self.df.shape[1] == 0
    
    async def auto_profile(self) -> DataProfile:
        """자동 데이터 프로파일링"""
        if self.profile is None:
            self.profile = DataProfile(
                shape=self.df.shape,
                dtypes={col: str(dtype) for col, dtype in self.df.dtypes.items()},
                missing_values=self.df.isnull().sum().to_dict(),
                memory_usage=self.df.memory_usage(deep=True).sum(),
                encoding=self.metadata.get('encoding', 'unknown'),
                file_size=self.metadata.get('file_size', 0)
            )
        return self.profile
    
    async def validate_quality(self) -> QualityReport:
        """데이터 품질 검증"""
        if self.quality_report is None:
            # 완전성 계산
            total_cells = self.df.shape[0] * self.df.shape[1]
            missing_cells = self.df.isnull().sum().sum()
            completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
            
            # 일관성 계산 (데이터 타입 일관성)
            consistency = 1.0  # 기본값, 더 정교한 로직 필요
            
            # 유효성 계산
            validity = 1.0  # 기본값, 더 정교한 로직 필요
            
            # 전체 점수
            overall_score = (completeness + consistency + validity) / 3
            
            issues = []
            recommendations = []
            
            if completeness < 0.9:
                issues.append(f"Missing values: {missing_cells} cells")
                recommendations.append("Consider imputation or removal of missing values")
            
            self.quality_report = QualityReport(
                overall_score=overall_score,
                completeness=completeness,
                consistency=consistency,
                validity=validity,
                issues=issues,
                recommendations=recommendations
            )
        
        return self.quality_report
```

### 5.2 A2A SDK 0.2.9 표준 적용

#### 5.2.1 표준 AgentExecutor 템플릿

```python
class StandardUnifiedAgentExecutor(AgentExecutor, UnifiedDataInterface):
    """A2A SDK 0.2.9 표준 + 통합 데이터 인터페이스"""
    
    def __init__(self, agent_name: str, specialized_config: Dict[str, Any]):
        super().__init__()
        self.agent_name = agent_name
        self.data_engine = LLMFirstDataEngine()
        self.cache_manager = CacheManager()
        self.specialized_config = specialized_config
        
        # 로깅 설정
        self.logger = logging.getLogger(f"UnifiedAgent.{agent_name}")
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """A2A 표준 실행 메서드"""
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 1단계: 사용자 쿼리 추출
            user_query = self._extract_user_query(context)
            self.logger.info(f"Processing query: {user_query}")
            
            await task_updater.update_status(
                TaskState.working,
                message=f"🧑🏻 {self.agent_name} 분석을 시작합니다..."
            )
            
            # 2단계: 의도 분석
            intent = await self.data_engine.analyze_intent(user_query, context)
            
            await task_updater.update_status(
                TaskState.working,
                message=f"🍒 의도 분석 완료: {intent.intent_type.value} (신뢰도: {intent.confidence:.2f})"
            )
            
            # 3단계: 데이터 로딩
            smart_df = await self.load_data(intent, context)
            
            await task_updater.update_status(
                TaskState.working,
                message=f"🍒 데이터 로딩 완료: {smart_df.shape[0]}행 x {smart_df.shape[1]}열"
            )
            
            # 4단계: 에이전트별 특화 처리
            results = await self._perform_specialized_processing(smart_df, intent, task_updater)
            
            # 5단계: 결과 반환 (A2A 표준)
            await self._send_a2a_results(results, task_updater)
            
            # 완료
            await task_updater.update_status(
                TaskState.completed,
                message="✅ 분석이 완료되었습니다."
            )
            
        except Exception as e:
            self.logger.error(f"Error in {self.agent_name}: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"❌ 오류가 발생했습니다: {str(e)}"
            )
    
    def _extract_user_query(self, context: RequestContext) -> str:
        """사용자 쿼리 추출 (A2A 표준)"""
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_query += part.root.text + " "
        return user_query.strip()
    
    async def load_data(self, intent: DataIntent, context) -> SmartDataFrame:
        """통합 데이터 로딩 구현"""
        # 캐시 확인
        cache_key = f"{self.agent_name}:{intent.intent_type.value}:{hash(str(intent.file_preferences))}"
        cached_df = await self.cache_manager.get(cache_key)
        
        if cached_df:
            self.logger.info("Using cached data")
            return cached_df
        
        # 파일 선택
        available_files = await self._scan_available_files()
        selected_file = await self.data_engine.select_optimal_file(intent, available_files)
        
        # 데이터 로딩
        smart_df = await self._load_file_safely(selected_file)
        
        # 캐싱
        await self.cache_manager.set(cache_key, smart_df, ttl=3600)
        
        return smart_df
    
    async def _send_a2a_results(self, results: Dict[str, Any], task_updater: TaskUpdater):
        """A2A 표준 결과 전송"""
        
        # 아티팩트 생성
        artifacts = []
        
        if 'dataframes' in results:
            for name, df in results['dataframes'].items():
                artifacts.append({
                    'name': f"{name}_data",
                    'type': 'dataframe', 
                    'data': df.to_json(),
                    'metadata': {'shape': df.shape}
                })
        
        if 'plots' in results:
            for plot_name, plot_data in results['plots'].items():
                artifacts.append({
                    'name': f"{plot_name}_plot",
                    'type': 'visualization',
                    'data': plot_data,
                    'metadata': {'format': 'plotly_json'}
                })
        
        # TextPart로 결과 전송 (A2A SDK 0.2.9 표준)
        for artifact in artifacts:
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(artifact))],
                name=artifact['name'],
                metadata=artifact['metadata']
            )
    
    @abstractmethod
    async def _perform_specialized_processing(self, smart_df: SmartDataFrame, intent: DataIntent, task_updater: TaskUpdater) -> Dict[str, Any]:
        """에이전트별 특화 처리 (하위 클래스에서 구현)"""
        pass
```

---

## 🧪 6. 테스트 및 검증 계획

### 6.1 테스트 전략

#### 6.1.1 단위 테스트 (pytest)
```bash
# 핵심 컴포넌트 단위 테스트
tests/unit/unified_data_loading/
├── test_unified_data_interface.py
├── test_llm_first_data_engine.py
├── test_smart_dataframe.py
├── test_cache_manager.py
└── test_file_connectors.py
```

#### 6.1.2 통합 테스트 (pytest)
```bash
# 에이전트별 통합 테스트
tests/integration/agent_migration/
├── test_data_loader_migration.py
├── test_data_cleaning_migration.py
├── test_data_visualization_migration.py
├── test_eda_tools_migration.py
└── test_complete_pipeline.py
```

#### 6.1.3 A2A 프로토콜 테스트 (pytest)
```bash
# A2A SDK 0.2.9 표준 준수 테스트
tests/a2a_compliance/
├── test_task_updater_pattern.py
├── test_text_part_serialization.py
├── test_artifact_generation.py
└── test_streaming_compliance.py
```

#### 6.1.4 최종 UI 테스트 (Playwright MCP)
```bash
# 실제 사용자 시나리오 테스트
tests/ui_validation/
├── test_file_upload_workflow.py
├── test_agent_collaboration.py
├── test_error_handling.py
└── test_performance_metrics.py
```

### 6.2 성능 벤치마크

#### 6.2.1 로딩 성능 테스트
```python
# 파일 크기별 로딩 성능 측정
test_files = {
    'small': '< 1MB',
    'medium': '1MB - 100MB', 
    'large': '100MB - 1GB',
    'extra_large': '> 1GB'
}

# 측정 지표
metrics = [
    'loading_time',
    'memory_usage',
    'cache_hit_ratio',
    'error_rate'
]
```

#### 6.2.2 에이전트별 성능 비교
- **Before**: 기존 개별 로딩 방식
- **After**: 통합 시스템 적용
- **Target**: 30% 성능 개선, 90% 에러 감소

### 6.3 기능 호환성 검증

#### 6.3.1 100% 기능 보존 체크리스트
```yaml
# 각 에이전트별 기능 체크리스트
data_loader:
  - ✅ CSV, Excel, JSON, Parquet 로딩
  - ✅ 인코딩 자동 감지
  - ✅ 대용량 파일 처리
  - ✅ 에러 복구 메커니즘

data_cleaning:
  - ✅ 결측값 처리
  - ✅ 이상값 탐지
  - ✅ 데이터 타입 최적화
  - ✅ 품질 보고서 생성

# ... 각 에이전트별 상세 체크리스트
```

---

## 📅 7. 구현 일정 및 마일스톤

### 7.1 전체 일정 (총 3주, 12개 에이전트 완전 마이그레이션)

#### Week 1: 핵심 인프라 + CRITICAL (7일)
- **Day 1-2**: 핵심 인프라 구축
  - UnifiedDataInterface + LLMFirstDataEngine 구현
  - SmartDataFrame + CacheManager 구현
- **Day 3-5**: CRITICAL 에이전트 (1개)
  - **data_loader (8307)** 완전 마이그레이션
  - 다중 구현체 통합, FileConnector 패턴 적용
- **Day 6-7**: CRITICAL 검증 및 HIGH 준비
  - data_loader 통합 테스트
  - HIGH 그룹 마이그레이션 준비

#### Week 2: HIGH 우선순위 에이전트 (7일)
- **Day 8-9**: HIGH 그룹 1차 (2개)
  - **data_cleaning (8306)**: 빈 데이터 오류 해결
  - **data_visualization (8308)**: UTF-8 인코딩 문제 해결
- **Day 10-11**: HIGH 그룹 2차 (2개)
  - **data_wrangling (8309)**: 파일 선택 안정성 개선
  - **eda_tools (8312)**: 통계 계산 오류 해결
- **Day 12-13**: HIGH 그룹 통합 테스트
  - 4개 에이전트 상호 호환성 검증
- **Day 14**: MEDIUM 그룹 마이그레이션 준비

#### Week 3: MEDIUM + LOW + 최종 검증 (7일)
- **Day 15-16**: MEDIUM 그룹 1차 (2개)
  - **feature_engineering (8310)**: 캐싱 시스템 도입
  - **sql_database (8311)**: DB 연결 안정성 개선
- **Day 17**: MEDIUM 그룹 2차 (2개)
  - **h2o_ml (8313)**: 모델링 안정성 개선
  - **mlflow_tools (8314)**: 실험 추적 안정성 개선
- **Day 18**: LOW 그룹 (2개)
  - **orchestrator (8100)**: 레지스트리 통합
  - **report_generator (8316)**: 종합 보고서 통합
- **Day 19**: 전체 시스템 통합 테스트 (12개 에이전트)
- **Day 20-21**: 최종 검증 및 문서화
  - Playwright MCP 완전 시나리오 테스트
  - **pandas_agent (8210)** 기준 모델 최종 검증

### 7.2 주요 마일스톤 (12개 에이전트 완전 추적)

| 마일스톤 | 일정 | 대상 에이전트 | 성공 기준 | 검증 방법 |
|---------|------|-------------|-----------|----------|
| **M1: 핵심 인프라** | Day 2 | 공통 인프라 | UnifiedDataInterface + LLMFirstDataEngine 완성 | 단위 테스트 100% 통과 |
| **M2: CRITICAL 완료** | Day 5 | data_loader (8307) | 다중 구현체 통합, 완벽 동작 | UI에서 파일 로딩 성공 |
| **M3: HIGH 1차 완료** | Day 9 | data_cleaning (8306)<br/>data_visualization (8308) | 빈 데이터 + 인코딩 문제 해결 | 차트 생성 성공 |
| **M4: HIGH 2차 완료** | Day 11 | data_wrangling (8309)<br/>eda_tools (8312) | 변환 + 통계 분석 안정화 | EDA 리포트 생성 성공 |
| **M5: HIGH 통합 완료** | Day 13 | HIGH 그룹 4개 | 상호 호환성 검증 완료 | 통합 워크플로우 성공 |
| **M6: MEDIUM 1차 완료** | Day 16 | feature_engineering (8310)<br/>sql_database (8311) | 캐싱 + DB 연결 안정성 | 특성 생성 + SQL 쿼리 성공 |
| **M7: MEDIUM 2차 완료** | Day 17 | h2o_ml (8313)<br/>mlflow_tools (8314) | ML 모델링 + 실험 추적 안정화 | AutoML + 실험 로깅 성공 |
| **M8: LOW 완료** | Day 18 | orchestrator (8100)<br/>report_generator (8316) | 조정 + 보고서 통합 완료 | 멀티 에이전트 조정 성공 |
| **M9: 전체 통합** | Day 19 | **12개 에이전트 전체** | 모든 에이전트 정상 동작 | 완전한 데이터 분석 파이프라인 |
| **M10: 최종 검증** | Day 21 | pandas_agent (8210) 기준 | 프로덕션 준비 완료 | Playwright 전체 시나리오 통과 |

**📊 진행률 추적:**
- **Week 1 완료**: 1/12 (8.3%) - CRITICAL
- **Week 2 완료**: 5/12 (41.7%) - CRITICAL + HIGH  
- **Week 3 완료**: 12/12 (100%) - 전체 시스템

---

## 🚀 8. 배포 및 모니터링

### 8.1 점진적 배포 전략

#### 8.1.1 단계별 배포
1. **Stage 1**: data_loader만 새 시스템 적용 (다른 에이전트는 기존 방식)
2. **Stage 2**: HIGH 우선순위 에이전트 추가
3. **Stage 3**: 전체 시스템 완전 전환

#### 8.1.2 롤백 계획
```bash
# 각 에이전트별 백업 보관
a2a_ds_servers/backup/
├── data_loader_original.py
├── data_cleaning_original.py
├── data_visualization_original.py
└── ... (전체 백업)

# 즉시 롤백 스크립트
./scripts/rollback_agent.sh [agent_name]
```

### 8.2 모니터링 시스템

#### 8.2.1 핵심 지표
```python
# 실시간 모니터링 지표
monitoring_metrics = {
    'data_loading': {
        'success_rate': '> 95%',
        'average_loading_time': '< 5초', 
        'cache_hit_ratio': '> 70%',
        'encoding_error_rate': '< 1%'
    },
    'agent_performance': {
        'response_time': '< 30초',
        'memory_usage': '< 1GB',
        'error_rate': '< 2%',
        'throughput': '> 10 requests/min'
    }
}
```

#### 8.2.2 알림 시스템
- **Critical**: 에이전트 다운, 데이터 로딩 실패 > 10%
- **Warning**: 성능 저하, 캐시 미스 증가
- **Info**: 정상 작동, 성능 개선 확인

---

## 📖 9. 문서화 및 가이드

### 9.1 개발자 가이드

#### 9.1.1 새 에이전트 추가 가이드
```python
# 새 에이전트 생성 템플릿
class NewAgentExecutor(StandardUnifiedAgentExecutor):
    def __init__(self):
        super().__init__(
            agent_name="NewAgent",
            specialized_config={
                "specific_feature": "value"
            }
        )
    
    async def _perform_specialized_processing(self, smart_df, intent, task_updater):
        # 에이전트별 특화 로직 구현
        results = await self._custom_analysis(smart_df, intent)
        return results
```

#### 9.1.2 트러블슈팅 가이드
```yaml
common_issues:
  utf8_encoding_error:
    symptoms: "'utf-8' codec can't decode"
    solution: "통합 인코딩 감지 시스템이 자동 해결"
    
  empty_dataframe_error:
    symptoms: "Cannot describe a DataFrame without columns"
    solution: "SmartDataFrame.is_empty() 검증 통과"
    
  file_selection_failure:
    symptoms: "No suitable file found"
    solution: "LLM 기반 지능형 파일 선택기 활용"
```

### 9.2 사용자 가이드

#### 9.2.1 데이터 업로드 가이드
- **지원 형식**: CSV, Excel (.xlsx/.xls), JSON, Parquet
- **권장 인코딩**: UTF-8
- **최대 파일 크기**: 1GB (자동 샘플링 적용)
- **업로드 위치**: `a2a_ds_servers/artifacts/data/shared_dataframes/`

#### 9.2.2 에이전트 사용법
각 에이전트별 최적 사용 패턴과 예시 쿼리 제공

---

## ✅ 10. 성공 기준 및 KPI

### 10.1 기술적 성공 기준

| 지표 | 현재 | 목표 | 측정 방법 |
|------|------|------|----------|
| **데이터 로딩 성공률** | ~70% | >95% | 테스트 실행 결과 |
| **인코딩 오류율** | ~20% | <1% | 에러 로그 분석 |
| **평균 로딩 시간** | ~15초 | <5초 | 성능 벤치마크 |
| **메모리 사용량** | 제한 없음 | <1GB | 시스템 모니터링 |
| **캐시 효율성** | 0% | >70% | 캐시 히트율 측정 |

### 10.2 사용자 경험 개선

| 항목 | 개선 목표 | 검증 방법 |
|------|-----------|----------|
| **에러 메시지** | 명확하고 해결책 포함 | 사용자 피드백 |
| **응답 속도** | 30초 이내 완료 | Playwright 테스트 |
| **안정성** | 24시간 무중단 운영 | 연속 실행 테스트 |
| **일관성** | 모든 에이전트 동일한 경험 | 크로스 에이전트 테스트 |

### 10.3 에이전트별 성공 기준 (12개 완전 추적)

#### 🔴 CRITICAL
- **data_loader (8307)**: 다중 구현체 → 단일 FileConnector 패턴, 로딩 성공률 95%+

#### 🟡 HIGH  
- **data_cleaning (8306)**: 빈 데이터 오류 0%, 컬럼 없음 에러 완전 해결
- **data_visualization (8308)**: UTF-8 인코딩 오류 0%, 차트 생성 성공률 98%+
- **data_wrangling (8309)**: 파일 선택 실패율 < 2%, 변환 오류 완전 해결
- **eda_tools (8312)**: 인코딩 문제 0%, 통계 계산 정확도 99%+

#### 🟢 MEDIUM
- **feature_engineering (8310)**: 캐시 히트율 70%+, 반복 로딩 시간 80% 단축
- **sql_database (8311)**: DB 연결 실패율 < 1%, 쿼리 실행 안정성 99%+
- **h2o_ml (8313)**: 모델링 실패율 < 5%, AutoML 성공률 95%+
- **mlflow_tools (8314)**: 실험 추적 손실율 < 1%, 로깅 완전성 99%+

#### 🔵 LOW
- **orchestrator (8100)**: 에이전트 발견율 100%, 조정 실패율 0%
- **report_generator (8316)**: 종합 분석 성공률 98%+, 다중 소스 통합 완성

#### ⭐ TEMPLATE
- **pandas_agent (8210)**: 모든 패턴의 기준, 완벽 동작 유지

### 10.4 전체 시스템 성공 기준

- **🎯 LLM First 원칙 100% 준수**: 12개 에이전트 모두 LLM이 동적으로 결정 수행
- **🔧 A2A SDK 0.2.9 완벽 적용**: 12개 에이전트 표준 프로토콜 완전 준수  
- **📊 기능 무손실 마이그레이션**: 12개 에이전트 기존 기능 100% 보존
- **🚀 성능 30% 개선**: 전체 시스템 로딩 속도 및 안정성 향상
- **🛠️ 유지보수성 향상**: 12개 에이전트 단일 코드베이스로 통합 관리
- **📈 전체 신뢰도 95%+**: 12개 에이전트 평균 신뢰도 목표 달성

---

## 🔚 결론

이 설계 문서는 CherryAI 시스템의 **12개 A2A 에이전트를 하나도 빠뜨리지 않고** pandas_agent의 우수한 패턴을 기준으로 통합하는 완전한 로드맵을 제공합니다.

### 📊 완전한 12개 에이전트 커버리지

**🔴 CRITICAL (1개)**: data_loader (8307)  
**🟡 HIGH (4개)**: data_cleaning (8306), data_visualization (8308), data_wrangling (8309), eda_tools (8312)  
**🟢 MEDIUM (4개)**: feature_engineering (8310), sql_database (8311), h2o_ml (8313), mlflow_tools (8314)  
**🔵 LOW (2개)**: orchestrator (8100), report_generator (8316)  
**⭐ TEMPLATE (1개)**: pandas_agent (8210)

### 🎯 핵심 원칙 (12개 에이전트 공통)

- ✅ **LLM First**: 12개 에이전트 모두 데이터 관련 결정을 LLM이 담당
- ✅ **A2A 표준**: 12개 에이전트 SDK 0.2.9 완벽 준수
- ✅ **기능 보존**: 12개 에이전트 Mock 없이 100% 실제 기능 유지
- ✅ **확장성**: 새로운 에이전트 쉽게 추가 가능한 통합 아키텍처

### 🚀 예상 효과 (전체 시스템)

- 🎯 **70% → 95%** 전체 시스템 데이터 로딩 성공률 향상
- ⚡ **15초 → 5초** 12개 에이전트 평균 응답 시간 단축  
- 🛡️ **20% → 1%** 전체 시스템 인코딩 오류율 감소
- 🧠 **100% LLM 기반** 12개 에이전트 지능형 데이터 처리
- 📈 **95%+ 신뢰도** 12개 에이전트 평균 분석 신뢰도 달성

### 🏆 최종 목표

이 계획에 따라 단계적으로 구현하면 **12개 A2A 에이전트가 완벽하게 통합된** 세계 최초의 완전한 LLM First + A2A 통합 데이터 과학 플랫폼을 완성할 수 있습니다.

**🍒 CherryAI = 12개 에이전트 + pandas_agent 패턴 + LLM First + A2A SDK 0.2.9** 