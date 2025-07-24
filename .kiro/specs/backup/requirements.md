# Requirements Document

## Introduction

CherryAI는 A2A SDK 0.2.9 기반의 멀티에이전트 데이터 분석 플랫폼으로, LLM First 철학을 준수하여 하드코딩 없이 범용적이면서도 전문적인 데이터 분석을 제공합니다. 

**현재 시스템 상태 (2025.01.19 기준):**
- ✅ A2A SDK 0.2.9 오케스트레이터 구현 완료 (a2a_orchestrator.py)
- ✅ 11개 A2A 에이전트 서버 구현 완료 (포트 8306-8316)
- ✅ Universal Engine 기반 cherry_ai.py 구현 완료
- ✅ start.sh/stop.sh 시스템 관리 스크립트 완료
- ⚠️ Langfuse v2 기본 구조 존재 (EMP_NO=2055186 통합 필요)
- ⚠️ SSE 스트리밍 기본 구현 (0.001초 지연 최적화 필요)
- ❌ E2E 검증 시스템 (Playwright MCP 통합 필요)
- ❌ 범용 도메인 적응 시스템 완전 검증 (LLM First Universal Engine)

**완성 목표:**
`streamlit run cherry_ai.py` 실행 → start.sh로 A2A 에이전트 구동 → 모든 에이전트 100% 동작 → Playwright MCP E2E 검증 → 범용 도메인 적응/일반인 시나리오 완전 지원 → qwen3-4b-fast 최적화로 ChatGPT Data Analyst 수준 달성

## Requirements

### Requirement 1

**User Story:** As a data analyst, I want to use CherryAI with streamlit run cherry_ai.py command, so that I can access a fully functional multi-agent data analysis platform

#### Acceptance Criteria

1. WHEN user runs `streamlit run cherry_ai.py` THEN system SHALL start successfully with all A2A agents operational
2. WHEN system starts THEN all 11 A2A agents SHALL be discoverable and functional on their designated ports (8306-8316)
3. WHEN system starts THEN orchestrator SHALL be running on port 8100 with full agent coordination capabilities

### Requirement 2

**User Story:** As a system administrator, I want to manage A2A agent servers using start.sh and stop.sh scripts, so that I can reliably control the entire system lifecycle

#### Acceptance Criteria

1. WHEN administrator runs `./start.sh` THEN system SHALL start all A2A agents in correct order with health checks
2. WHEN administrator runs `./stop.sh` THEN system SHALL gracefully shutdown all agents and clean up resources
3. WHEN agents are starting THEN system SHALL verify each agent is ready before proceeding to next agent
4. WHEN system encounters startup failures THEN system SHALL provide clear error messages and cleanup procedures

### Requirement 3

**User Story:** As a developer, I want the system to fully comply with A2A SDK 0.2.9 standards, so that all agents work seamlessly with the latest protocol specifications

#### Acceptance Criteria

1. WHEN agents are implemented THEN they SHALL use A2A SDK 0.2.9 import patterns and API methods
2. WHEN agents communicate THEN they SHALL follow A2A protocol specifications for message passing
3. WHEN agents are discovered THEN they SHALL provide proper agent cards with capabilities and skills
4. WHEN tasks are executed THEN they SHALL use proper A2A task management and state handling

### Requirement 4

**User Story:** As a data scientist, I want all agent functionalities to be 100% operational and E2E verified, so that I can trust the system for professional data analysis work

#### Acceptance Criteria

1. WHEN each agent receives requests THEN it SHALL process them correctly and return accurate results
2. WHEN agents collaborate THEN they SHALL coordinate effectively through the orchestrator
3. WHEN system is tested THEN all agent functions SHALL pass comprehensive E2E verification tests
4. WHEN edge cases occur THEN agents SHALL handle errors gracefully and provide meaningful feedback

### Requirement 5

**User Story:** As a user, I want comprehensive scenario verification using Playwright MCP, so that I can be confident the system works for both novice and expert use cases

#### Acceptance Criteria

1. WHEN novice user scenarios are tested THEN system SHALL provide intuitive guidance and clear results
2. WHEN expert scenarios are tested with ion_implant_3lot_dataset.csv THEN system SHALL demonstrate universal domain adaptation by automatically detecting semiconductor characteristics and providing expert-level analysis
3. WHEN query.txt content is processed THEN system SHALL demonstrate LLM-based dynamic domain expertise without hardcoded knowledge
4. WHEN Playwright MCP tests run THEN they SHALL verify complete user workflows from data upload to final analysis, proving universal domain adaptability

### Requirement 6

**User Story:** As a system architect, I want the system to follow LLM First principles with zero hardcoding, intelligent self-critique & replanning, and smart query routing, so that it can adapt universally to any data analysis scenario with optimal efficiency and quality

#### Acceptance Criteria

1. WHEN system processes requests THEN it SHALL NOT use rule-based hardcoding or pattern matching
2. WHEN system analyzes query complexity THEN it SHALL use LLM-based unified analysis instead of hardcoded thresholds
3. WHEN system performs self-critique THEN it SHALL separate critique (evaluation) and replanning (strategy) roles to avoid bias
4. WHEN system determines processing strategy THEN it SHALL use LLM to decide between fast_track|balanced|thorough|expert_mode approaches
5. WHEN trivial queries are received THEN system SHALL provide direct responses without orchestrator overhead (5-10초)
6. WHEN simple queries are received THEN system SHALL use single-agent processing for efficiency (10-20초)
7. WHEN complex queries are received THEN system SHALL use multi-agent orchestration for comprehensive analysis (30-60초)
8. WHEN multi-agent collaboration occurs THEN it SHALL be clearly visible and demonstrate intelligent coordination with LLM-guided orchestration

### Requirement 6.1 - Smart Query Routing System

**User Story:** As a user, I want the system to intelligently route my queries based on complexity analysis, so that I get optimal response time and quality for each type of request

#### Acceptance Criteria

1. WHEN query is received THEN system SHALL perform LLM-based quick complexity assessment to determine routing strategy
2. WHEN trivial queries are detected THEN system SHALL route to direct response mode bypassing orchestrator (5-10초)
3. WHEN simple queries are detected THEN system SHALL route to single-agent processing mode (10-20초)
4. WHEN complex queries are detected THEN system SHALL route to full multi-agent orchestration mode (30-60초)
5. WHEN routing decision is made THEN system SHALL provide clear indication of processing mode to user
6. WHEN single-agent mode is used THEN system SHALL select most appropriate agent using LLM-based analysis
7. WHEN routing occurs THEN all modes SHALL maintain proper Langfuse session tracking
8. WHEN routing strategy changes THEN system SHALL adapt dynamically based on query characteristics

### Requirement 7

**User Story:** As a performance-conscious user, I want the system optimized for local Ollama qwen3-4b-fast model, so that I can get balanced speed and quality in local environment

#### Acceptance Criteria

1. WHEN system uses LLM THEN it SHALL be configured for OLLAMA_MODEL=qwen3-4b-fast from .env settings
2. WHEN analysis is performed THEN system SHALL provide accurate numerical results without hallucination
3. WHEN responses are generated THEN they SHALL be trustworthy with proper data-driven insights
4. WHEN system operates THEN it SHALL balance local processing speed with analysis quality

### Requirement 8

**User Story:** As a user, I want real-time streaming responses with SSE, so that I can see analysis progress and results as they develop

#### Acceptance Criteria

1. WHEN analysis starts THEN system SHALL provide SSE-based async streaming with chunk-based updates
2. WHEN streaming occurs THEN it SHALL include appropriate delays (0.001s) for smooth user experience
3. WHEN multi-agent processing happens THEN streaming SHALL show collaborative progress clearly
4. WHEN analysis completes THEN final results SHALL be presented in comprehensive, ChatGPT-style format

### Requirement 9

**User Story:** As a data analyst, I want comprehensive analysis reports with code and insights, so that I can understand both the process and results like ChatGPT Data Analyst

#### Acceptance Criteria

1. WHEN analysis is requested THEN system SHALL provide meaningful analysis reports with supporting code
2. WHEN user requests match their intent THEN system SHALL deliver precisely what was asked for
3. WHEN numerical data is presented THEN it SHALL be accurate and provide reliability indicators
4. WHEN data quality is assessed THEN system SHALL evaluate it from a data scientist perspective

### Requirement 10

**User Story:** As a system monitor, I want comprehensive Langfuse v2 logging with proper session tracking, so that I can trace and analyze system performance

#### Acceptance Criteria

1. WHEN user interactions occur THEN they SHALL be logged to Langfuse v2 with EMP_NO=2055186 as user ID
2. WHEN multi-agent analysis happens THEN it SHALL be traced as single cohesive session
3. WHEN sessions are created THEN they SHALL be properly tracked and organized for analysis
4. WHEN logging occurs THEN it SHALL capture complete workflow from user question through final response

### Requirement 11

**User Story:** As a domain expert in any field, I want the system to automatically adapt to my domain and provide expert-level analysis, so that I can get professional insights regardless of my field

#### Acceptance Criteria

1. WHEN any domain-specific dataset is uploaded THEN system SHALL automatically detect domain characteristics using LLM-based analysis
2. WHEN complex domain queries are submitted THEN system SHALL demonstrate adaptive expert-level knowledge through LLM reasoning
3. WHEN domain anomalies are detected THEN system SHALL provide technical analysis and recommendations based on LLM understanding
4. WHEN domain-specific metrics are calculated THEN they SHALL be accurate and professionally relevant through dynamic LLM processing

### Requirement 12

**User Story:** As a developer, I want A2A SDK 0.2.9 fully implemented with proper import patterns, so that the system uses the latest protocol features correctly

#### Acceptance Criteria

1. WHEN agents are implemented THEN they SHALL use `from a2a.server.apps import A2AStarletteApplication` instead of deprecated patterns
2. WHEN request handlers are created THEN they SHALL use `from a2a.server.request_handlers import DefaultRequestHandler` with proper initialization
3. WHEN agent execution occurs THEN it SHALL use `from a2a.server.agent_execution import AgentExecutor, RequestContext` with correct async patterns
4. WHEN agent cards are defined THEN they SHALL include proper `AgentCapabilities(streaming=True)` configuration

### Requirement 13

**User Story:** As a system monitor, I want Langfuse v2.60.8 session-based tracing fully operational, so that I can track complete multi-agent workflows

#### Acceptance Criteria

1. WHEN user sessions start THEN they SHALL use `SessionBasedTracer` with session ID format `user_query_{timestamp}_{user_id}`
2. WHEN agents execute THEN they SHALL be wrapped with `LangfuseEnhancedA2AExecutor` for automatic tracing
3. WHEN streaming occurs THEN trace context SHALL be maintained throughout `RealTimeStreamingTaskUpdater` operations
4. WHEN sessions end THEN they SHALL include comprehensive metadata with agent performance metrics

### Requirement 14

**User Story:** As a user, I want optimized SSE streaming with proper chunk delays, so that I can see smooth real-time analysis progress

#### Acceptance Criteria

1. WHEN streaming starts THEN system SHALL implement `async def process_query_streaming()` with proper SSE headers
2. WHEN chunks are sent THEN they SHALL include 0.001s delays using `await asyncio.sleep(0.001)` for smooth UX
3. WHEN multi-agent coordination occurs THEN streaming SHALL show `async for chunk_data in client.stream_task()` progress
4. WHEN streaming completes THEN final response SHALL be accumulated and presented as complete analysis

### Requirement 15

**User Story:** As a QA engineer, I want Playwright MCP integration for comprehensive E2E testing, so that I can verify complete user scenarios

#### Acceptance Criteria

1. WHEN E2E tests run THEN they SHALL use `from playwright.async_api import async_playwright` for browser automation
2. WHEN UI interactions are tested THEN they SHALL verify Streamlit components with `await page.wait_for_selector('[data-testid="stApp"]')`
3. WHEN file uploads are tested THEN they SHALL simulate real CSV file uploads and verify processing
4. WHEN analysis workflows are tested THEN they SHALL capture screenshots and verify multi-agent collaboration

### Requirement 16

**User Story:** As a system architect, I want MCP server integration properly configured, so that hybrid A2A+MCP architecture works seamlessly

#### Acceptance Criteria

1. WHEN MCP servers start THEN they SHALL use `MCPConfigManager` with proper port configuration (8006-8020)
2. WHEN MCP tools are discovered THEN they SHALL include domain-agnostic tools that can adapt to any domain through LLM reasoning
3. WHEN hybrid workflows execute THEN they SHALL coordinate between A2A agents (8306-8316) and MCP tools effectively
4. WHEN MCP health checks run THEN they SHALL verify all enabled servers are responsive and functional

### Requirement 17

**User Story:** As a performance engineer, I want Ollama qwen3-4b-fast optimization with proper LLM factory integration, so that local processing is efficient

#### Acceptance Criteria

1. WHEN LLM instances are created THEN they SHALL use `create_llm_instance()` with automatic Langfuse callback injection
2. WHEN Ollama is configured THEN it SHALL use `OLLAMA_BASE_URL=http://localhost:11434` with `qwen3-4b-fast` model
3. WHEN LLM calls are made THEN they SHALL include proper temperature and token limit settings for data analysis
4. WHEN performance is measured THEN response times SHALL be optimized for local environment constraints

### Requirement 18

**User Story:** As a data scientist, I want comprehensive error handling and recovery mechanisms, so that the system is robust and reliable

#### Acceptance Criteria

1. WHEN agent failures occur THEN system SHALL implement graceful fallback strategies with error logging
2. WHEN network timeouts happen THEN system SHALL retry with exponential backoff and user notification
3. WHEN data processing errors occur THEN system SHALL provide meaningful error messages with recovery suggestions
4. WHEN system health degrades THEN monitoring SHALL trigger alerts and automatic recovery procedures

### Requirement 19

**User Story:** As a QA engineer, I want to test the system's universal domain adaptation using complex semiconductor data (ion_implant_3lot_dataset.csv + query.txt), so that I can verify the LLM First Universal Engine can handle expert-level domain analysis without hardcoded knowledge

#### Acceptance Criteria

1. WHEN ion_implant_3lot_dataset.csv is uploaded THEN system SHALL automatically detect domain characteristics using LLM-based analysis without hardcoded semiconductor patterns
2. WHEN query.txt content (complex domain knowledge query) is submitted THEN system SHALL demonstrate LLM-based dynamic domain expertise adaptation
3. WHEN complex domain analysis is performed THEN system SHALL provide accurate analysis through pure LLM reasoning without predefined domain logic
4. WHEN analysis results are provided THEN they SHALL demonstrate the system's ability to achieve expert-level insights through universal domain adaptation

### Requirement 20

**User Story:** As a general user, I want ChatGPT-style intuitive UI/UX with drag-and-drop file upload, so that I can easily perform data analysis without technical expertise

#### Acceptance Criteria

1. WHEN user accesses cherry_ai.py THEN system SHALL provide ChatGPT-like chat interface with file upload area
2. WHEN user drags and drops files THEN system SHALL accept CSV/Excel files with visual feedback
3. WHEN analysis is running THEN system SHALL show real-time progress with agent collaboration visualization
4. WHEN results are ready THEN system SHALL present them in comprehensive report format with code, charts, and insights

### Requirement 21

**User Story:** As a system integrator, I want all 11 existing A2A agents to have 100% of their implemented functions verified and validated, so that every existing capability is guaranteed to work correctly

#### Acceptance Criteria

1. WHEN each of 11 existing agents (data_cleaning, data_loader, data_visualization, data_wrangling, feature_engineering, sql_database, eda_tools, h2o_ml, mlflow_tools, pandas_analyst, report_generator) is tested THEN all their currently implemented functions SHALL pass 100% verification
2. WHEN agent function inventory is performed THEN system SHALL discover and catalog all existing functions in each agent
3. WHEN function verification tests are executed THEN each discovered function SHALL be tested with appropriate test cases
4. WHEN verification results are generated THEN they SHALL provide detailed pass/fail status for every function in every agent
5. WHEN agents collaborate through orchestrator THEN all inter-agent communications SHALL work flawlessly with existing functions
6. WHEN complex multi-agent workflows are executed THEN they SHALL complete successfully using verified functions
7. WHEN agent health checks are performed THEN all agents SHALL respond correctly on their designated ports with all functions operational

### Requirement 21.1 - Data Cleaning Agent (Port 8306) 완전 검증

**User Story:** As a data quality engineer, I want data cleaning agent to perform all cleaning operations flawlessly, so that data quality is guaranteed

#### Acceptance Criteria

1. WHEN missing value detection is requested THEN agent SHALL identify all null, NaN, empty string values with accurate counts
2. WHEN missing value handling is requested THEN agent SHALL provide multiple strategies (drop, fill mean/median/mode, forward/backward fill, interpolation)
3. WHEN outlier detection is requested THEN agent SHALL identify outliers using IQR, Z-score, isolation forest methods
4. WHEN outlier treatment is requested THEN agent SHALL provide removal, capping, transformation options
5. WHEN data type validation is requested THEN agent SHALL detect and correct inappropriate data types
6. WHEN duplicate detection is requested THEN agent SHALL identify exact and fuzzy duplicates
7. WHEN data standardization is requested THEN agent SHALL normalize text, dates, categorical values
8. WHEN data validation rules are applied THEN agent SHALL check constraints, ranges, patterns

### Requirement 21.2 - Data Loader Agent (Port 8307) 완전 검증

**User Story:** As a data engineer, I want data loader agent to handle all data sources and formats perfectly, so that any data can be imported

#### Acceptance Criteria

1. WHEN CSV files are uploaded THEN agent SHALL handle various encodings (UTF-8, CP949, etc.) and delimiters
2. WHEN Excel files are uploaded THEN agent SHALL read multiple sheets, handle merged cells, and preserve formatting
3. WHEN JSON files are uploaded THEN agent SHALL parse nested structures and flatten if needed
4. WHEN database connections are requested THEN agent SHALL connect to MySQL, PostgreSQL, SQLite, SQL Server
5. WHEN large files are processed THEN agent SHALL use chunking and streaming for memory efficiency
6. WHEN file parsing errors occur THEN agent SHALL provide detailed error messages and recovery suggestions
7. WHEN data preview is requested THEN agent SHALL show sample data with column info and statistics
8. WHEN data schema inference is needed THEN agent SHALL automatically detect column types and suggest corrections

### Requirement 21.3 - Data Visualization Agent (Port 8308) 완전 검증

**User Story:** As a data analyst, I want visualization agent to create all types of charts and plots perfectly, so that data insights are clearly communicated

#### Acceptance Criteria

1. WHEN basic plots are requested THEN agent SHALL create line, bar, scatter, histogram, box plots with proper formatting
2. WHEN advanced plots are requested THEN agent SHALL create heatmaps, violin plots, pair plots, correlation matrices
3. WHEN interactive plots are requested THEN agent SHALL create Plotly charts with zoom, hover, selection features
4. WHEN statistical plots are requested THEN agent SHALL create distribution plots, Q-Q plots, regression plots
5. WHEN time series plots are requested THEN agent SHALL handle datetime axes, seasonal decomposition, trend analysis
6. WHEN multi-dimensional plots are requested THEN agent SHALL create 3D plots, subplots, faceted charts
7. WHEN custom styling is requested THEN agent SHALL apply themes, colors, annotations, titles, legends
8. WHEN plot export is requested THEN agent SHALL save in PNG, SVG, HTML, PDF formats

### Requirement 21.4 - Data Wrangling Agent (Port 8309) 완전 검증

**User Story:** As a data scientist, I want data wrangling agent to perform all data transformation operations perfectly, so that data is analysis-ready

#### Acceptance Criteria

1. WHEN data filtering is requested THEN agent SHALL apply complex conditions, multiple criteria, date ranges
2. WHEN data sorting is requested THEN agent SHALL sort by single/multiple columns, custom orders, null handling
3. WHEN data grouping is requested THEN agent SHALL group by categories, time periods, custom functions
4. WHEN data aggregation is requested THEN agent SHALL compute sum, mean, count, min, max, percentiles, custom functions
5. WHEN data merging is requested THEN agent SHALL perform inner, outer, left, right joins with key matching
6. WHEN data reshaping is requested THEN agent SHALL pivot, melt, transpose, stack, unstack operations
7. WHEN data sampling is requested THEN agent SHALL perform random, stratified, systematic sampling
8. WHEN data splitting is requested THEN agent SHALL create train/test/validation splits with proper ratios

### Requirement 21.5 - Feature Engineering Agent (Port 8310) 완전 검증

**User Story:** As a machine learning engineer, I want feature engineering agent to create and select features perfectly, so that model performance is optimized

#### Acceptance Criteria

1. WHEN numerical feature creation is requested THEN agent SHALL create polynomial, interaction, ratio, log features
2. WHEN categorical feature encoding is requested THEN agent SHALL perform one-hot, label, target, binary encoding
3. WHEN text feature extraction is requested THEN agent SHALL create TF-IDF, word counts, n-grams, embeddings
4. WHEN datetime feature extraction is requested THEN agent SHALL extract year, month, day, hour, weekday, season
5. WHEN feature scaling is requested THEN agent SHALL apply standardization, normalization, robust scaling
6. WHEN feature selection is requested THEN agent SHALL use correlation, mutual info, chi-square, recursive elimination
7. WHEN dimensionality reduction is requested THEN agent SHALL apply PCA, t-SNE, UMAP, factor analysis
8. WHEN feature importance is requested THEN agent SHALL calculate permutation, SHAP, tree-based importance

### Requirement 21.6 - SQL Database Agent (Port 8311) 완전 검증

**User Story:** As a database analyst, I want SQL database agent to execute all database operations perfectly, so that database analysis is comprehensive

#### Acceptance Criteria

1. WHEN database connection is requested THEN agent SHALL connect to multiple database types with proper authentication
2. WHEN SQL queries are executed THEN agent SHALL handle SELECT, INSERT, UPDATE, DELETE operations safely
3. WHEN complex queries are requested THEN agent SHALL create JOINs, subqueries, CTEs, window functions
4. WHEN query optimization is requested THEN agent SHALL suggest indexes, query rewrites, execution plans
5. WHEN database schema analysis is requested THEN agent SHALL describe tables, columns, relationships, constraints
6. WHEN data profiling is requested THEN agent SHALL analyze distributions, cardinality, null rates, patterns
7. WHEN query results are processed THEN agent SHALL handle large result sets with pagination and streaming
8. WHEN database errors occur THEN agent SHALL provide meaningful error messages and recovery suggestions

### Requirement 21.7 - EDA Tools Agent (Port 8312) 완전 검증

**User Story:** As a data scientist, I want EDA tools agent to perform comprehensive exploratory analysis perfectly, so that data insights are thoroughly discovered

#### Acceptance Criteria

1. WHEN descriptive statistics are requested THEN agent SHALL compute mean, median, mode, std, skewness, kurtosis
2. WHEN correlation analysis is requested THEN agent SHALL calculate Pearson, Spearman, Kendall correlations with significance
3. WHEN distribution analysis is requested THEN agent SHALL test normality, fit distributions, create Q-Q plots
4. WHEN categorical analysis is requested THEN agent SHALL create frequency tables, chi-square tests, Cramér's V
5. WHEN time series analysis is requested THEN agent SHALL detect trends, seasonality, stationarity, autocorrelation
6. WHEN anomaly detection is requested THEN agent SHALL identify outliers, change points, unusual patterns
7. WHEN data quality assessment is requested THEN agent SHALL check completeness, consistency, validity, uniqueness
8. WHEN automated insights are requested THEN agent SHALL generate narrative summaries of key findings

### Requirement 21.8 - H2O ML Agent (Port 8313) 완전 검증

**User Story:** As a machine learning practitioner, I want H2O ML agent to perform all machine learning operations perfectly, so that model development is comprehensive

#### Acceptance Criteria

1. WHEN AutoML is requested THEN agent SHALL train multiple algorithms and select best model automatically
2. WHEN classification models are requested THEN agent SHALL train Random Forest, GBM, XGBoost, Neural Networks
3. WHEN regression models are requested THEN agent SHALL train Linear, GLM, GBM, Deep Learning models
4. WHEN model evaluation is requested THEN agent SHALL compute accuracy, precision, recall, F1, AUC, RMSE, MAE
5. WHEN hyperparameter tuning is requested THEN agent SHALL perform grid search, random search, Bayesian optimization
6. WHEN feature importance is requested THEN agent SHALL provide SHAP values, permutation importance, variable importance
7. WHEN model interpretation is requested THEN agent SHALL create partial dependence plots, LIME explanations
8. WHEN model deployment is requested THEN agent SHALL export models in MOJO, POJO, pickle formats

### Requirement 21.9 - MLflow Tools Agent (Port 8314) 완전 검증

**User Story:** As an MLOps engineer, I want MLflow tools agent to manage ML lifecycle perfectly, so that experiments and models are properly tracked

#### Acceptance Criteria

1. WHEN experiment tracking is requested THEN agent SHALL log parameters, metrics, artifacts, model versions
2. WHEN model registry is accessed THEN agent SHALL register, version, stage, and transition models
3. WHEN model serving is requested THEN agent SHALL deploy models as REST APIs with proper endpoints
4. WHEN experiment comparison is requested THEN agent SHALL compare runs, metrics, parameters across experiments
5. WHEN artifact management is requested THEN agent SHALL store and retrieve datasets, models, plots, reports
6. WHEN model monitoring is requested THEN agent SHALL track model performance, drift, data quality
7. WHEN pipeline orchestration is requested THEN agent SHALL create and manage ML workflows
8. WHEN collaboration features are used THEN agent SHALL support team access, permissions, sharing

### Requirement 21.10 - Pandas Analyst Agent (Port 8210) 완전 검증

**User Story:** As a data analyst, I want pandas analyst agent to perform all pandas operations perfectly, so that data manipulation and analysis is comprehensive

#### Acceptance Criteria

1. WHEN data loading is requested THEN agent SHALL read various formats with proper parsing options
2. WHEN data inspection is requested THEN agent SHALL provide info, describe, head, tail, shape, dtypes
3. WHEN data selection is requested THEN agent SHALL filter rows, select columns, slice data with complex conditions
4. WHEN data manipulation is requested THEN agent SHALL apply, map, transform, replace, rename operations
5. WHEN data aggregation is requested THEN agent SHALL groupby, pivot_table, crosstab, resample operations
6. WHEN data merging is requested THEN agent SHALL merge, join, concat, append with various strategies
7. WHEN data cleaning is requested THEN agent SHALL handle missing values, duplicates, data types
8. WHEN statistical analysis is requested THEN agent SHALL compute correlations, distributions, hypothesis tests

### Requirement 21.11 - Report Generator Agent (Port 8316) 완전 검증

**User Story:** As a business analyst, I want report generator agent to create comprehensive reports perfectly, so that analysis results are professionally presented

#### Acceptance Criteria

1. WHEN executive summary is requested THEN agent SHALL create high-level insights with key findings
2. WHEN detailed analysis report is requested THEN agent SHALL include methodology, results, conclusions
3. WHEN data quality report is requested THEN agent SHALL assess completeness, accuracy, consistency
4. WHEN statistical report is requested THEN agent SHALL include descriptive stats, tests, confidence intervals
5. WHEN visualization report is requested THEN agent SHALL embed charts, tables, interactive elements
6. WHEN comparative analysis is requested THEN agent SHALL compare datasets, time periods, segments
7. WHEN recommendation report is requested THEN agent SHALL provide actionable insights and next steps
8. WHEN export functionality is requested THEN agent SHALL generate PDF, HTML, Word, PowerPoint formats

### Requirement 22

**User Story:** As a QA engineer, I want Playwright MCP-based comprehensive E2E testing for both novice and expert scenarios, so that I can verify complete system functionality

#### Acceptance Criteria

1. WHEN novice scenario E2E test runs THEN it SHALL simulate simple data upload and basic analysis request
2. WHEN expert scenario E2E test runs THEN it SHALL upload ion_implant_3lot_dataset.csv and process query.txt content
3. WHEN Playwright MCP automation runs THEN it SHALL capture screenshots and verify UI interactions
4. WHEN E2E tests complete THEN they SHALL generate comprehensive test reports with pass/fail status for all scenarios

### Requirement 23

**User Story:** As a performance engineer, I want qwen3-4b-fast model optimization with balanced speed and quality, so that local environment provides practical usability

#### Acceptance Criteria

1. WHEN LLM calls are made THEN system SHALL use optimized prompts and parameters for qwen3-4b-fast model
2. WHEN multi-agent processing occurs THEN system SHALL balance concurrent LLM usage for optimal throughput
3. WHEN analysis is performed THEN response time SHALL be practical for interactive use while maintaining quality
4. WHEN system resources are monitored THEN memory and CPU usage SHALL be optimized for local deployment

### Requirement 24

**User Story:** As a system monitor, I want EMP_NO=2055186 integrated Langfuse v2 logging with complete session tracking, so that I can trace every user interaction and agent collaboration

#### Acceptance Criteria

1. WHEN user starts analysis THEN system SHALL create Langfuse session with EMP_NO=2055186 as user_id
2. WHEN multi-agent workflow executes THEN all agent activities SHALL be traced under single session
3. WHEN session completes THEN Langfuse SHALL contain complete trace from user query to final response
4. WHEN session data is reviewed THEN it SHALL include performance metrics, agent contributions, and execution timeline