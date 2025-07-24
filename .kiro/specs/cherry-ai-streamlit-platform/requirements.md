# Requirements Document

## Introduction

Cherry AI Streamlit Platform은 검증된 LLM First Universal Engine 패턴을 기반으로 한 멀티 에이전트 데이터 사이언스 플랫폼입니다. ChatGPT/Claude와 같은 직관적인 사용자 경험을 제공하면서도 A2A SDK 0.2.9 기반의 에이전트 간 협업 과정을 투명하게 시각화하는 신뢰할 수 있는 데이터 분석 플랫폼을 Streamlit으로 구현합니다.

핵심 철학은 **검증된 LLM First 패턴 활용**으로, 이미 100% 구현 완료된 Universal Engine의 성공 패턴들을 재사용하여:
- **MetaReasoningEngine**: 4단계 추론 (초기 관찰 → 다각도 분석 → 자가 검증 → 적응적 응답)
- **A2AAgentDiscoverySystem**: 검증된 에이전트 발견 및 헬스 모니터링
- **A2AWorkflowOrchestrator**: 검증된 순차/병렬 실행 패턴
- **Progressive Disclosure**: 검증된 사용자 적응형 UI 패턴

이를 통해 하드코딩된 규칙 없이 LLM 추론 능력을 활용하여 다양한 데이터와 분석 요구사항에 동적으로 대응합니다.

## Requirements

### Requirement 1: ChatGPT/Claude 스타일 단일 페이지 채팅 인터페이스

**User Story:** As a data analyst, I want a single-page chat interface similar to ChatGPT/Claude with enhanced UI/UX features, so that I can interact with the platform naturally and intuitively.

#### Acceptance Criteria

1. WHEN the user accesses the platform THEN the system SHALL display a single main page with all functionality integrated in vertical layout
2. WHEN the user views the interface THEN the system SHALL show enhanced ChatGPT/Claude style with drag-and-drop file upload area at top, scrollable chat interface in center, and message input with keyboard shortcuts at bottom
3. WHEN the user types a message THEN the system SHALL support Shift+Enter for line breaks, Enter for sending, auto-resize text area, and placeholder text "여기에 메시지를 입력하세요..."
4. WHEN the user sends a message THEN the system SHALL display user messages right-aligned with distinct speech bubbles and AI responses left-aligned with system avatar and Cherry AI branding
5. WHEN the system processes requests THEN the system SHALL show real-time typing indicators, progress bars, and agent collaboration visualization during work
6. WHEN messages are displayed THEN the system SHALL provide auto-scroll to bottom, session persistence across browser refresh, and optional timestamp display
7. WHEN files are attached THEN the system SHALL show 📎 attachment button, file preview, and immediate processing status with visual feedback

### Requirement 2: 향상된 파일 처리 및 데이터 카드 시스템

**User Story:** As a user, I want an intuitive file upload experience with visual data cards and immediate analysis suggestions, so that I can start analyzing data immediately with clear visual feedback.

#### Acceptance Criteria

1. WHEN the user uploads a file THEN the system SHALL provide drag-and-drop area with clear visual boundaries, upload progress indicators, and automatic processing by Pandas Analyst Agent (8315)
2. WHEN files are uploaded THEN the system SHALL support CSV, Excel (.xlsx, .xls), JSON, Parquet, and PKL formats with multi-file selection and real-time upload status
3. WHEN multiple files are uploaded THEN the system SHALL display visual data cards with dataset name, size (rows×columns), memory usage, preview button (top 10 rows), and selection checkboxes (default: all selected)
4. WHEN data is processed THEN the system SHALL automatically detect format, perform quality validation, generate basic profiling, and show immediate visual feedback
5. WHEN data relationships exist THEN the system SHALL display visual relationship diagrams showing connection possibilities based on schema similarity and common columns
6. WHEN analysis suggestions are generated THEN the system SHALL provide one-click execution buttons with clear action descriptions and estimated completion time
7. WHEN single dataset is uploaded THEN the system SHALL suggest contextual analysis (time series for temporal data, statistical for numerical, frequency for categorical) with visual preview
8. WHEN multiple datasets are uploaded THEN the system SHALL analyze inter-dataset relationships, suggest merge possibilities, and propose comparative analysis with relationship visualization

### Requirement 3: A2A SDK 0.2.9 에이전트 통합 및 협업 (검증된 Universal Engine 패턴 기반)

**User Story:** As a data scientist, I want seamless integration with all A2A agents using proven Universal Engine patterns and transparent collaboration visualization, so that I can trust the analysis process and understand how results are generated.

#### Acceptance Criteria

1. WHEN the system operates THEN the system SHALL integrate with all 10 A2A agents using validated A2AAgentDiscoverySystem patterns including Data Cleaning (8306), Data Loader (8307), Data Visualization (8308), Data Wrangling (8309), Feature Engineering (8310), SQL Database (8311), EDA Tools (8312), H2O ML (8313), MLflow Tools (8314), and Pandas Collaboration Hub (8315)
2. WHEN communicating with agents THEN the system SHALL use proven A2ACommunicationProtocol with JSON-RPC 2.0 and /.well-known/agent.json validation patterns
3. WHEN agents execute THEN the system SHALL implement validated enhanced_agent_request patterns with user_expertise_level, domain_context, and collaboration_mode
4. WHEN analysis is requested THEN the system SHALL use proven LLMBasedAgentSelector for dynamic agent selection without hardcoded rules
5. WHEN agents collaborate THEN the system SHALL use validated A2AWorkflowOrchestrator with sequential/parallel execution patterns and real-time progress visualization
6. WHEN agents complete tasks THEN the system SHALL use proven A2AResultIntegrator for conflict resolution and unified insights presentation
7. WHEN agent failures occur THEN the system SHALL implement validated A2AErrorHandler with progressive retry (5s → 15s → 30s) and circuit breaker patterns

### Requirement 4: 실시간 에이전트 협업 시각화 및 스트리밍 시스템

**User Story:** As a user, I want to see real-time agent collaboration with visual progress indicators and natural streaming responses, so that I can understand what's happening and trust the analysis process.

#### Acceptance Criteria

1. WHEN agents execute THEN the system SHALL display real-time collaboration visualization with progress bars (0-100%), agent avatars, status messages, and completion checkmarks with execution time
2. WHEN the system processes requests THEN the system SHALL use A2A SDK 0.2.9 TaskUpdater pattern with visual feedback for submit() → start_work() → update_status() lifecycle
3. WHEN responses stream THEN the system SHALL implement natural typing effects with 0.001 second delays, intelligent chunking by semantic units, and natural pauses at punctuation marks
4. WHEN agent status updates THEN the system SHALL show visual indicators for TaskState.working (spinning icon), TaskState.completed (checkmark), and TaskState.failed (error icon) with color coding
5. WHEN multiple agents work THEN the system SHALL display concurrent execution with individual progress tracking, agent-specific status messages, and data flow visualization between agents
6. WHEN streaming is interrupted THEN the system SHALL implement graceful degradation with clear error messages and fallback to basic responses
7. WHEN agents collaborate THEN the system SHALL show real-time updates every 0.5 seconds with agent names, current tasks, and inter-agent data transfer status

### Requirement 5: 향상된 A2A 아티팩트 렌더링 시스템

**User Story:** As a data analyst, I want beautiful, interactive rendering of all analysis artifacts with enhanced UX features, so that I can easily interpret and interact with visualizations, tables, and reports.

#### Acceptance Criteria

1. WHEN agents generate Plotly charts THEN the system SHALL render fully interactive charts with hover effects, zoom/pan functionality, responsive container sizing, and Streamlit theme integration
2. WHEN agents generate images THEN the system SHALL display PNG, JPG, SVG files with automatic size adjustment, Base64 encoding support, PIL Image formats, and click-to-enlarge functionality
3. WHEN agents generate text artifacts THEN the system SHALL render markdown with complete support for tables, lists, code blocks, LaTeX math formulas, and safe HTML rendering with XSS prevention
4. WHEN agents generate code THEN the system SHALL apply syntax highlighting for Python, SQL, JSON with line numbers, copy-to-clipboard functionality, and language-specific formatting
5. WHEN agents generate data tables THEN the system SHALL implement virtual scrolling for performance, column sorting/filtering, conditional formatting with color coding, and statistical summaries for numerical columns
6. WHEN artifacts are received THEN the system SHALL automatically route to appropriate renderers based on metadata with fallback to JSON display for unknown types and error handling for rendering failures
7. WHEN large datasets are displayed THEN the system SHALL provide pagination controls, search functionality, and export options (CSV, Excel) with download progress indicators

### Requirement 6: 멀티 데이터 관리 시스템

**User Story:** As a data scientist working with multiple datasets, I want intelligent data relationship discovery and management, so that I can easily work with complex multi-dataset scenarios.

#### Acceptance Criteria

1. WHEN multiple datasets are uploaded THEN the system SHALL display data cards with selection checkboxes (default: all selected)
2. WHEN datasets are analyzed THEN the system SHALL automatically discover relationships based on schema similarity and common columns
3. WHEN relationships are found THEN the system SHALL display visual relationship diagrams showing connection possibilities
4. WHEN working with data THEN the system SHALL track current working dataset context and maintain session state
5. WHEN data selection changes THEN the system SHALL update analysis suggestions based on selected datasets

### Requirement 7: 지능적 추천 시스템 (검증된 Universal Engine 패턴 기반)

**User Story:** As a user, I want intelligent analysis recommendations based on my data and previous results using proven LLM First patterns, so that I can discover insights without needing deep analytical expertise.

#### Acceptance Criteria

1. WHEN analysis completes THEN the system SHALL generate maximum 3 follow-up analysis recommendations using validated DynamicContextDiscovery patterns
2. WHEN recommendations are shown THEN the system SHALL provide one-click execution buttons for immediate analysis start
3. WHEN files are uploaded THEN the system SHALL use proven zero-hardcoding data understanding that automatically discovers domain context without predefined categories
4. WHEN user patterns are available THEN the system SHALL use validated RealTimeLearningSystem patterns for suggestion improvement based on user feedback
5. WHEN recommendations are generated THEN the system SHALL use proven AdaptiveUserUnderstanding to estimate user expertise and adapt suggestions accordingly

### Requirement 8: Progressive Disclosure 결과 표시 및 스마트 다운로드 시스템

**User Story:** As a data analyst, I want clear result presentation with progressive disclosure and intelligent download options that provide raw artifacts plus context-aware enhanced formats, so that I can easily understand insights and access data in the most appropriate format.

#### Acceptance Criteria

1. WHEN analysis completes THEN the system SHALL display summary results first showing 3-5 key insights with visual highlights, important metrics emphasis, and bullet-point format for easy scanning
2. WHEN detailed view is requested THEN the system SHALL provide "📄 View All Details" expandable button revealing complete analysis results, agent-specific work history, executed code with syntax highlighting, intermediate results, and inter-agent data transfer records
3. WHEN artifacts are generated THEN the system SHALL always provide raw artifact downloads (Chart Data JSON, Table Data CSV, Code PY, Image PNG) as the primary download options that are always available
4. WHEN download options are displayed THEN the system SHALL show context-aware enhanced formats: chart images (PNG/SVG/HTML) for visualizations, Excel/PDF for business users, Jupyter notebooks for developers, and complete reports for comprehensive analysis
5. WHEN results are displayed THEN the system SHALL use progressive disclosure with collapsible sections, visual hierarchy, and clear section headers for easy navigation
6. WHEN detailed panel is opened THEN the system SHALL provide smooth toggle functionality for expand/collapse with animation effects and state persistence
7. WHEN download options are accessed THEN the system SHALL show two-tier download system: "Raw Artifacts (Always Available)" section with original A2A agent outputs, and "Enhanced Formats (Context-Based)" section with user-appropriate additional options
8. WHEN sharing results THEN the system SHALL provide shareable links, embedded code snippets, and export templates optimized for different stakeholder types (technical/non-technical) with appropriate format recommendations

### Requirement 9: 모듈화된 아키텍처 (검증된 Universal Engine 패턴 기반)

**User Story:** As a developer, I want a clean, modular architecture based on proven Universal Engine patterns, so that the system is maintainable and extensible.

#### Acceptance Criteria

1. WHEN the system is structured THEN app.py SHALL be kept under 50 lines using validated CherryAIUniversalEngineUI patterns
2. WHEN modules are organized THEN the system SHALL separate functionality into modules/core/, modules/ui/, modules/data/, modules/artifacts/, modules/a2a/, and modules/utils/ following proven Universal Engine structure
3. WHEN A2A SDK is used THEN the system SHALL follow validated A2A integration patterns including A2AClient, A2AWorkflowOrchestrator, and A2AResultIntegrator
4. WHEN new agents are added THEN the system SHALL use proven UniversalEngineInitializer for automatic discovery and capability mapping
5. WHEN code is written THEN the system SHALL follow validated Universal Engine patterns with complete type hints and detailed docstrings

### Requirement 10: 성능 및 안정성 (검증된 Universal Engine 패턴 기반)

**User Story:** As a user, I want reliable performance with large datasets and multiple concurrent operations using proven Universal Engine patterns, so that I can work efficiently without system limitations.

#### Acceptance Criteria

1. WHEN processing large datasets THEN the system SHALL handle 10MB files in under 10 seconds with memory usage under 1GB per session using validated performance optimization patterns
2. WHEN multiple users access THEN the system SHALL support 50 concurrent users without performance degradation using proven SessionManager patterns
3. WHEN errors occur THEN the system SHALL maintain 99.9% uptime with automatic recovery within 5 seconds using validated A2AErrorHandler patterns
4. WHEN agents fail THEN the system SHALL implement proven health checks, progressive retry (5s → 15s → 30s), and circuit breaker patterns
5. WHEN files are uploaded THEN the system SHALL use validated security patterns for malicious code scanning and secure temporary file management

### Requirement 11: 향상된 사용자 경험 및 접근성 최적화

**User Story:** As a non-technical user, I want an intuitive interface with comprehensive visual feedback and accessibility features, so that I can perform data analysis efficiently without extensive training.

#### Acceptance Criteria

1. WHEN first-time users access THEN the system SHALL enable completion of first analysis within 5 minutes through intuitive workflow: drag-and-drop upload → automatic suggestions → one-click execution → clear results
2. WHEN users perform actions THEN the system SHALL provide immediate visual feedback with loading states, progress indicators, success animations, and clear status messages for every interaction
3. WHEN errors occur THEN the system SHALL display user-friendly messages with actionable guidance, converting technical errors to understandable language, and providing recovery suggestions
4. WHEN workflows are designed THEN the system SHALL follow ChatGPT/Claude patterns with familiar UI elements, consistent visual language, and predictable interaction patterns
5. WHEN help is needed THEN the system SHALL provide contextual tooltips, inline guidance, placeholder text, and self-explanatory UI elements minimizing external documentation needs
6. WHEN accessibility is required THEN the system SHALL support keyboard navigation, screen reader compatibility, high contrast mode, and responsive design for various screen sizes
7. WHEN user feedback is collected THEN the system SHALL provide satisfaction ratings, usage analytics, and continuous UX improvement based on user behavior patterns

### Requirement 12: 확장성 및 배포 (검증된 Universal Engine 패턴 기반)

**User Story:** As a system administrator, I want easy deployment and scaling options using proven Universal Engine patterns, so that I can efficiently manage the platform in production environments.

#### Acceptance Criteria

1. WHEN deploying THEN the system SHALL provide optimized Docker image with docker-compose configuration using validated Universal Engine deployment patterns
2. WHEN scaling is needed THEN the system SHALL support horizontal scaling with external state storage and agent pooling using proven patterns
3. WHEN monitoring is required THEN the system SHALL use validated PerformanceMonitoringSystem with structured JSON logging and health check endpoints
4. WHEN configuration is needed THEN the system SHALL use proven LLMFactory patterns with environment variables (LLM_PROVIDER=OLLAMA, OLLAMA_MODEL)
5. WHEN new agents are added THEN the system SHALL use validated A2AAgentDiscoverySystem for automatic discovery and registration without system restart

### Requirement 13: 원클릭 실행 및 분석 추천 시스템

**User Story:** As a user, I want intelligent analysis recommendations with one-click execution buttons, so that I can easily discover and execute relevant analyses without deep technical knowledge.

#### Acceptance Criteria

1. WHEN analysis completes THEN the system SHALL generate maximum 3 contextual follow-up recommendations with clear descriptions, estimated completion time, and complexity indicators
2. WHEN recommendations are displayed THEN the system SHALL provide prominent one-click execution buttons with visual icons, action descriptions, and expected outcomes
3. WHEN files are uploaded THEN the system SHALL immediately suggest relevant analyses based on data characteristics with preview of expected results
4. WHEN multiple datasets are available THEN the system SHALL suggest relationship analyses, merge operations, and comparative studies with visual relationship indicators
5. WHEN user patterns are detected THEN the system SHALL learn from previous analysis choices and personalize future recommendations
6. WHEN recommendations are executed THEN the system SHALL provide clear feedback on execution status, progress indicators, and seamless transition to results

### Requirement 14: 반응형 디자인 및 브라우저 호환성

**User Story:** As a user accessing from different devices and browsers, I want consistent and optimized experience across all platforms, so that I can work efficiently regardless of my setup.

#### Acceptance Criteria

1. WHEN accessing from different screen sizes THEN the system SHALL provide responsive design with adaptive layouts, touch-friendly controls, and optimized mobile experience
2. WHEN charts and tables are displayed THEN the system SHALL automatically adjust container sizes, provide horizontal scrolling for large tables, and maintain aspect ratios for visualizations
3. WHEN using different browsers THEN the system SHALL ensure consistent functionality across Chrome, Firefox, Safari, and Edge with graceful degradation for unsupported features
4. WHEN touch interfaces are used THEN the system SHALL provide appropriate button sizes, gesture support, and mobile-optimized file upload experience
5. WHEN network conditions vary THEN the system SHALL implement progressive loading, offline capability indicators, and bandwidth-adaptive content delivery