# CherryAI 통합 시스템 요구사항 명세서

## 📋 개요

이 문서는 **검증 완료된 LLM-First Universal Engine 백엔드**와 **ChatGPT 스타일 UI/UX 개선사항**을 통합한 CherryAI 시스템의 요구사항을 정의합니다.

### 현재 상황 분석
- **✅ 백엔드 완성**: LLM-First Universal Engine (100% 검증 완료)
  - 26개 컴포넌트, 19개 메서드 구현 완료
  - Zero-hardcoding 달성 (31개 위반 사항 해결)
  - A2A SDK 0.2.9 완전 통합
  - qwen3-4b-fast 최적화 (평균 45초 응답)
- **⚠️ 프론트엔드 개선 필요**: cherry_ai.py UI/UX 향상
- **❌ 누락 기능**: Langfuse v2, SSE 스트리밍, E2E 검증

### 통합 목표
1. **백엔드 보존**: 검증된 Universal Engine 100% 유지
2. **UI/UX 개선**: ChatGPT 스타일 인터페이스 구현
3. **기능 완성**: Langfuse v2, 스트리밍, E2E 테스트 추가
4. **전문가 시나리오**: 이온주입 데이터 + query.txt 완벽 지원

## Requirements

### Requirement 1: 검증된 백엔드 시스템 보존

**User Story:** As a system architect, I want to preserve the fully verified LLM-First Universal Engine backend, so that all existing functionality remains stable and reliable.

#### Acceptance Criteria

1. WHEN the integrated system starts THEN it SHALL use the existing Universal Engine without any modifications to core logic
2. WHEN data analysis is performed THEN it SHALL utilize the verified 26 components and 19 implemented methods
3. WHEN LLM processing occurs THEN it SHALL maintain the zero-hardcoding architecture with 100% LLM-based decisions
4. WHEN A2A agents are orchestrated THEN it SHALL use the existing A2A SDK 0.2.9 integration patterns
5. WHEN performance is measured THEN it SHALL maintain the optimized qwen3-4b-fast response times (45 second average)

### Requirement 2: ChatGPT 스타일 UI/UX 통합

**User Story:** As an end user, I want a ChatGPT-style interface that seamlessly integrates with the Universal Engine backend, so that I can perform data analysis with familiar and intuitive interactions.

#### Acceptance Criteria

1. WHEN I access cherry_ai.py THEN the system SHALL display a ChatGPT-style chat interface powered by the Universal Engine
2. WHEN I upload data files THEN the system SHALL use the existing DataManager and Universal Engine for processing
3. WHEN analysis is performed THEN the system SHALL show real-time agent collaboration through the verified orchestration system
4. WHEN results are displayed THEN the system SHALL present them in ChatGPT-style format with code, visualizations, and explanations
5. WHEN I ask follow-up questions THEN the system SHALL maintain conversation context using the Universal Engine's adaptive understanding

### Requirement 3: A2A 에이전트 시스템 완전 통합 및 100% 기능 검증

**User Story:** As a system administrator, I want all 12 A2A agents to work seamlessly with the new UI while maintaining the verified backend orchestration, so that multi-agent collaboration is transparent and reliable.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL initialize all 12 A2A agents (ports 8306-8316) using start.sh
2. WHEN agents are orchestrated THEN it SHALL use the verified A2AAgentDiscoverySystem and A2AWorkflowOrchestrator
3. WHEN agent status is checked THEN the UI SHALL display real-time status using the existing monitoring components
4. WHEN agents collaborate THEN the system SHALL show the collaboration process through the Universal Engine's transparency features
5. WHEN agents fail THEN the system SHALL use the verified error handling and recovery mechanisms

### Requirement 3.1: 전체 에이전트 기능 100% 검증

**User Story:** As a quality assurance engineer, I want every single function of all 12 A2A agents to be 100% verified and tested, so that I can guarantee complete system reliability.

#### Acceptance Criteria

1. WHEN comprehensive testing is performed THEN it SHALL verify all 88+ functions across all 12 A2A agents individually
2. WHEN each agent is tested THEN it SHALL validate every endpoint, method, and capability with specific test cases
3. WHEN function verification occurs THEN it SHALL include:
   - **Data Cleaning Agent (Port 8306)**: All 8 cleaning functions (missing values, outliers, duplicates, etc.)
   - **Data Loader Agent (Port 8307)**: All 6 loading functions (CSV, Excel, JSON, database connections, etc.)
   - **Data Visualization Agent (Port 8308)**: All 10 visualization functions (plots, charts, interactive visualizations, etc.)
   - **Data Wrangling Agent (Port 8309)**: All 7 wrangling functions (transformations, aggregations, joins, etc.)
   - **Feature Engineering Agent (Port 8310)**: All 9 feature functions (creation, selection, scaling, encoding, etc.)
   - **SQL Database Agent (Port 8311)**: All 8 SQL functions (queries, connections, schema operations, etc.)
   - **EDA Tools Agent (Port 8312)**: All 12 EDA functions (statistical analysis, correlation, distribution analysis, etc.)
   - **H2O ML Agent (Port 8313)**: All 10 ML functions (model training, evaluation, prediction, AutoML, etc.)
   - **MLflow Tools Agent (Port 8314)**: All 8 MLflow functions (experiment tracking, model registry, deployment, etc.)
   - **Python REPL Agent (Port 8315)**: All 6 execution functions (code execution, environment management, etc.)
   - **Pandas Analyst Agent (Port 8316)**: All 12 analysis functions (data manipulation, statistical analysis, etc.)
   - **Orchestrator Agent**: All 8 orchestration functions (workflow management, agent coordination, etc.)
4. WHEN function testing is complete THEN it SHALL generate detailed verification reports for each agent with pass/fail status for every function
5. WHEN any function fails THEN the system SHALL provide specific error details, root cause analysis, and remediation steps

### Requirement 4: Langfuse v2 세션 기반 추적 시스템

**User Story:** As a system monitor, I want comprehensive Langfuse v2 logging with proper session tracking, so that I can trace and analyze system performance with EMP_NO=2055186 as user ID.

#### Acceptance Criteria

1. WHEN user interactions occur THEN they SHALL be logged to Langfuse v2 with EMP_NO=2055186 as user_id
2. WHEN multi-agent analysis happens THEN it SHALL be traced as single cohesive session with format `user_query_{timestamp}_{user_id}`
3. WHEN Universal Engine components execute THEN they SHALL be wrapped with LangfuseEnhancedA2AExecutor for automatic tracing
4. WHEN sessions are created THEN they SHALL be properly tracked and organized for analysis
5. WHEN sessions end THEN they SHALL include comprehensive metadata with agent performance metrics

### Requirement 5: SSE 실시간 스트리밍 시스템

**User Story:** As an end user, I want smooth real-time streaming of analysis results with minimal delay, so that I can see progress and results as they happen.

#### Acceptance Criteria

1. WHEN analysis starts THEN the system SHALL provide SSE streaming with 0.001 second delay intervals
2. WHEN Universal Engine processes queries THEN it SHALL stream intermediate results through RealTimeStreamingTaskUpdater
3. WHEN agents execute THEN their progress SHALL be streamed in real-time with chunk-based updates
4. WHEN results are generated THEN they SHALL be streamed progressively without waiting for completion
5. WHEN streaming occurs THEN trace context SHALL be maintained throughout the entire process

### Requirement 6: 전문가 시나리오 완벽 지원

**User Story:** As a semiconductor expert, I want to upload ion_implant_3lot_dataset.csv and query with query.txt content to get professional-level analysis, so that I can make informed engineering decisions.

#### Acceptance Criteria

1. WHEN I upload ion_implant_3lot_dataset.csv THEN the Universal Engine SHALL automatically detect semiconductor domain context
2. WHEN I input query.txt content THEN the system SHALL understand the complex domain knowledge and evaluation criteria
3. WHEN analysis is performed THEN the system SHALL provide accurate numerical results with proper domain interpretation
4. WHEN results are presented THEN they SHALL include professional-level insights suitable for 20-year experienced engineers
5. WHEN anomalies are detected THEN the system SHALL provide specific technical recommendations and action plans

### Requirement 7: E2E 검증 시스템 (Playwright MCP)

**User Story:** As a quality assurance engineer, I want comprehensive E2E testing using Playwright MCP integration, so that I can verify all functionality works correctly in real scenarios.

#### Acceptance Criteria

1. WHEN E2E tests run THEN they SHALL use Playwright MCP for browser automation and testing
2. WHEN general user scenarios are tested THEN they SHALL cover basic data analysis workflows
3. WHEN expert scenarios are tested THEN they SHALL specifically test ion_implant_3lot_dataset.csv with query.txt content
4. WHEN multi-agent collaboration is tested THEN it SHALL verify all 12 agents work correctly together
5. WHEN streaming is tested THEN it SHALL verify SSE functionality and real-time updates work properly

### Requirement 8: LLM First 최적화 및 성능

**User Story:** As a performance engineer, I want the system to maintain LLM First principles while achieving optimal performance with qwen3-4b-fast, so that users get fast and accurate results.

#### Acceptance Criteria

1. WHEN LLM instances are created THEN they SHALL use the verified LLMFactory with automatic Langfuse callback injection
2. WHEN Ollama is configured THEN it SHALL use OLLAMA_BASE_URL=http://localhost:11434 with qwen3-4b-fast model
3. WHEN LLM calls are made THEN they SHALL include proper temperature and token limit settings for data analysis
4. WHEN performance is measured THEN it SHALL maintain the verified 45-second average response time
5. WHEN quality is assessed THEN it SHALL achieve 0.8/1.0 quality score while maintaining speed

### Requirement 9: 시스템 관리 및 운영

**User Story:** As a system administrator, I want reliable system management through start.sh/stop.sh scripts, so that I can easily manage the entire CherryAI system.

#### Acceptance Criteria

1. WHEN I run start.sh THEN it SHALL start all 12 A2A agents and the orchestrator in correct order
2. WHEN I run stop.sh THEN it SHALL gracefully shutdown all components and clean up resources
3. WHEN agents are monitored THEN the system SHALL provide real-time health checks and status updates
4. WHEN errors occur THEN the system SHALL use the verified error handling and recovery mechanisms
5. WHEN system is restarted THEN it SHALL restore all previous state and continue operations seamlessly

### Requirement 10: 사용자 경험 최적화

**User Story:** As a data analyst, I want intelligent recommendations and seamless interactions, so that I can efficiently perform complex data analysis tasks.

#### Acceptance Criteria

1. WHEN I upload data THEN the system SHALL provide up to 3 intelligent analysis recommendations using the Universal Engine
2. WHEN analysis completes THEN the system SHALL suggest relevant follow-up analyses based on results
3. WHEN I interact with the system THEN it SHALL adapt to my expertise level using AdaptiveUserUnderstanding
4. WHEN visualizations are created THEN they SHALL be interactive and exportable in multiple formats
5. WHEN code is generated THEN it SHALL be displayed with proper syntax highlighting and explanations

### Requirement 11: 도메인 적응성 및 확장성

**User Story:** As a domain expert, I want the system to automatically adapt to different domains while maintaining the Universal Engine's flexibility, so that I can analyze any type of data effectively.

#### Acceptance Criteria

1. WHEN different domain data is uploaded THEN the Universal Engine SHALL automatically detect domain characteristics
2. WHEN domain-specific analysis is needed THEN the system SHALL select appropriate agents and methodologies
3. WHEN new domains are encountered THEN the system SHALL adapt without requiring code changes
4. WHEN complex domain knowledge is required THEN the system SHALL leverage the MetaReasoningEngine for deep analysis
5. WHEN domain expertise varies THEN the system SHALL adjust explanations and recommendations accordingly

### Requirement 12: 품질 보증 및 신뢰성

**User Story:** As a quality manager, I want comprehensive quality assurance mechanisms, so that all analysis results are accurate and trustworthy.

#### Acceptance Criteria

1. WHEN analysis is performed THEN the system SHALL use the verified quality assessment mechanisms
2. WHEN results are generated THEN they SHALL be validated through the MetaReasoningEngine's quality evaluation
3. WHEN errors are detected THEN the system SHALL provide clear explanations and recovery options
4. WHEN data quality issues exist THEN the system SHALL identify and report them with recommendations
5. WHEN system performance degrades THEN it SHALL automatically optimize and maintain quality standards

### Requirement 13: 통합 테스트 및 검증

**User Story:** As a test engineer, I want comprehensive integration testing that verifies the UI works perfectly with the verified backend, so that the entire system functions reliably.

#### Acceptance Criteria

1. WHEN integration tests run THEN they SHALL verify UI components work with Universal Engine backend
2. WHEN component tests execute THEN they SHALL confirm all 26 components and 19 methods function correctly
3. WHEN performance tests run THEN they SHALL validate the 45-second response time target is maintained
4. WHEN E2E scenarios execute THEN they SHALL test both general and expert use cases comprehensively
5. WHEN regression tests run THEN they SHALL ensure no existing functionality is broken by UI changes

## 🎯 성공 기준

### 기능적 완성도
- **백엔드 보존**: 검증된 Universal Engine 100% 유지
- **UI/UX 통합**: ChatGPT 스타일 인터페이스 완전 구현
- **기능 완성**: Langfuse v2, SSE 스트리밍, E2E 테스트 100% 구현

### 성능 및 품질
- **응답 시간**: 평균 45초 유지 (검증된 성능)
- **품질 점수**: 0.8/1.0 달성
- **시스템 안정성**: 99.9% 가용성
- **사용자 만족도**: ChatGPT 수준의 UX 제공

### 전문가 시나리오 지원
- **도메인 분석**: 반도체 이온주입 공정 완벽 분석
- **전문 해석**: 20년 경력 엔지니어 수준의 인사이트 제공
- **실무 적용**: 구체적인 조치 방안 및 기술적 권장사항 제공

이 통합 요구사항을 충족하면 검증된 백엔드의 안정성을 유지하면서 최고 수준의 사용자 경험을 제공하는 완전한 CherryAI 시스템이 구축됩니다.