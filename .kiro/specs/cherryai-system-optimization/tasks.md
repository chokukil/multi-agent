# CherryAI 시스템 최적화 구현 계획

## 📋 개요

이 구현 계획은 기존에 구현된 11개 A2A 에이전트의 모든 기능을 검증하고, LLM First 원칙 기반의 최적화된 시스템을 구축하는 것을 목표로 합니다. 핵심은 기존 기능의 100% 검증과 ChatGPT Data Analyst 수준의 사용자 경험 달성입니다.

## Phase 1: 기존 에이전트 기능 검증 시스템 구현

### 1.1 ExistingAgentFunctionValidator 핵심 시스템 구현

- [ ] 1.1.1 ExistingAgentFunctionValidator 클래스 구현
  - ExistingAgentFunctionValidator 클래스 정의 및 초기화
  - AgentFunctionDiscovery, ComprehensiveFunctionTester, ValidationReporter 의존성 주입
  - discover_and_validate_all_agents() 메서드 구현
  - 11개 에이전트 순차 검증 로직 구현
  - _Requirements: 21.1, 21.2, 21.3_

- [ ] 1.1.2 AgentFunctionDiscovery 시스템 구현
  - AgentFunctionDiscovery 클래스 정의 및 초기화
  - discover_agent_functions() 메서드 구현
  - A2A 클라이언트 연결 및 에이전트 정보 수집
  - 에이전트 카드에서 기능 목록 추출 로직
  - _Requirements: 21.2, 21.3_

- [ ] 1.1.3 구현 파일 기반 기능 발견 시스템
  - discover_from_implementation() 메서드 구현
  - a2a_ds_servers/{agent_name}_server.py 파일 읽기 및 분석
  - 정규표현식 기반 함수 정의 패턴 매칭
  - private 함수 제외 및 공개 함수만 추출
  - _Requirements: 21.1, 21.2_

- [ ] 1.1.4 API 엔드포인트 기반 기능 발견 시스템
  - discover_from_api_endpoints() 메서드 구현
  - 각 에이전트 포트별 API 엔드포인트 스캔
  - 동적 기능 발견 및 메타데이터 수집
  - 기능 정보 통합 및 중복 제거
  - _Requirements: 21.3, 21.4_

### 1.2 ComprehensiveFunctionTester 구현

- [ ] 1.2.1 ComprehensiveFunctionTester 클래스 구현
  - ComprehensiveFunctionTester 클래스 정의 및 초기화
  - test_function() 메서드 구현
  - 5단계 테스트 프로세스 구현 (연결, 파라미터, 실행, 에러처리, 성능)
  - 테스트 결과 구조화 및 상태 판정 로직
  - _Requirements: 21.1, 21.2, 21.3_

- [ ] 1.2.2 기본 연결 테스트 시스템
  - test_basic_connection() 메서드 구현
  - 에이전트 포트 매핑 시스템 (8306-8316, 8210)
  - aiohttp 기반 health check 엔드포인트 테스트
  - HTTP 상태 코드 검증 및 응답성 확인
  - _Requirements: 21.1, 21.2_

- [ ] 1.2.3 파라미터 검증 테스트 시스템
  - test_parameters() 메서드 구현
  - 함수 파라미터 정보 기반 검증 테스트
  - 필수/선택 파라미터 구분 및 타입 검증
  - 파라미터 범위 및 제약조건 테스트
  - _Requirements: 21.2, 21.3_

- [ ] 1.2.4 실제 기능 실행 테스트 시스템
  - test_function_execution() 메서드 구현
  - prepare_test_data() 에이전트별 테스트 데이터 준비
  - call_agent_function() A2A 클라이언트 기반 기능 호출
  - 실행 결과 검증 및 오류 감지 로직
  - _Requirements: 21.1, 21.2, 21.3_

- [ ] 1.2.5 에러 처리 테스트 시스템
  - test_error_handling() 메서드 구현
  - 의도적 오류 상황 생성 및 테스트
  - 에러 메시지 품질 및 복구 가능성 검증
  - 예외 처리 로직 견고성 테스트
  - _Requirements: 18.1, 18.2, 18.3_

- [ ] 1.2.6 성능 테스트 시스템
  - test_performance() 메서드 구현
  - 응답 시간 측정 및 성능 메트릭 수집
  - 메모리 사용량 및 리소스 효율성 평가
  - 성능 기준 대비 평가 및 최적화 권장사항
  - _Requirements: 17.4, 23.2_

### 1.3 ValidationReporter 구현

- [ ] 1.3.1 ValidationReporter 클래스 구현
  - ValidationReporter 클래스 정의 및 초기화
  - generate_comprehensive_report() 메서드 구현
  - 검증 결과 통계 계산 (총 에이전트, 기능, 성공률)
  - JSON 형태 종합 리포트 생성 및 파일 저장
  - _Requirements: 21.1, 21.2, 21.3_

- [ ] 1.3.2 권장사항 생성 시스템
  - generate_recommendations() 메서드 구현
  - 실패한 기능들에 대한 분석 및 개선 방안 제시
  - 성능 최적화 권장사항 생성
  - 에이전트별 우선순위 개선 항목 도출
  - _Requirements: 21.1, 21.2_

- [ ] 1.3.3 다음 단계 계획 생성 시스템
  - generate_next_steps() 메서드 구현
  - 검증 결과 기반 후속 작업 계획 수립
  - 실패 기능 수정 우선순위 및 일정 제안
  - 시스템 안정성 향상을 위한 액션 플랜
  - _Requirements: 21.1, 21.2_

## Phase 2: 11개 에이전트 기존 기능 완전 검증

### 2.1 Data Cleaning Agent (포트 8306) 기능 검증

- [ ] 2.1.1 누락값 감지 기능 검증
  - detect_missing_values() 메서드 동작 검증
  - null, NaN, 빈 문자열 식별 정확성 테스트
  - 정확한 카운트 및 백분율 계산 검증
  - 컬럼별 누락값 패턴 분석 결과 확인
  - _Requirements: 21.1.1_

- [ ] 2.1.2 누락값 처리 기능 검증
  - handle_missing_values() 메서드 동작 검증
  - 다중 전략 (drop, mean, median, mode, forward/backward fill, interpolation) 테스트
  - 전략별 처리 결과 정확성 검증
  - 처리 전후 데이터 무결성 확인
  - _Requirements: 21.1.2_

- [ ] 2.1.3 이상치 감지 기능 검증
  - detect_outliers() 메서드 동작 검증
  - IQR, Z-score, Isolation Forest 방법 정확성 테스트
  - 다중 방법 결과 일관성 검증
  - 이상치 시각화 출력 확인
  - _Requirements: 21.1.3_

- [ ] 2.1.4 이상치 처리 기능 검증
  - treat_outliers() 메서드 동작 검증
  - 제거, 캡핑, 변환 옵션 정확성 테스트
  - 처리 전후 비교 결과 검증
  - 데이터 무결성 보장 확인
  - _Requirements: 21.1.4_

- [ ] 2.1.5 데이터 타입 검증 기능 검증
  - validate_data_types() 메서드 동작 검증
  - 부적절한 데이터 타입 감지 정확성 테스트
  - 타입 변환 제안 적절성 검증
  - 변환 안전성 확인
  - _Requirements: 21.1.5_

- [ ] 2.1.6 중복 데이터 감지 기능 검증
  - detect_duplicates() 메서드 동작 검증
  - 정확한 중복 및 퍼지 중복 감지 정확성 테스트
  - 중복 패턴 분석 결과 검증
  - 중복 제거 옵션 동작 확인
  - _Requirements: 21.1.6_

- [ ] 2.1.7 데이터 표준화 기능 검증
  - standardize_data() 메서드 동작 검증
  - 텍스트, 날짜, 범주형 값 정규화 정확성 테스트
  - 표준화 규칙 커스터마이징 동작 검증
  - 표준화 품질 결과 확인
  - _Requirements: 21.1.7_

- [ ] 2.1.8 데이터 검증 규칙 적용 기능 검증
  - apply_validation_rules() 메서드 동작 검증
  - 제약조건, 범위, 패턴 검증 정확성 테스트
  - 커스텀 검증 규칙 지원 확인
  - 검증 결과 리포팅 품질 검증
  - _Requirements: 21.1.8_

### 2.2 Data Loader Agent (포트 8307) 기능 검증

- [ ] 2.2.1 CSV 파일 로딩 기능 검증
  - load_csv_files() 메서드 동작 검증
  - 다양한 인코딩 (UTF-8, CP949, etc.) 지원 테스트
  - 구분자 자동 감지 및 커스터마이징 검증
  - 대용량 CSV 파일 처리 성능 확인
  - _Requirements: 21.2.1_

- [ ] 2.2.2 Excel 파일 로딩 기능 검증
  - load_excel_files() 메서드 동작 검증
  - 다중 시트 처리 및 선택 기능 테스트
  - 병합된 셀 처리 정확성 검증
  - 서식 정보 보존 확인
  - _Requirements: 21.2.2_

- [ ] 2.2.3 JSON 파일 로딩 기능 검증
  - load_json_files() 메서드 동작 검증
  - 중첩 구조 파싱 및 평면화 정확성 테스트
  - 복잡한 JSON 구조 처리 검증
  - 스키마 추론 및 검증 결과 확인
  - _Requirements: 21.2.3_

- [ ] 2.2.4 데이터베이스 연결 기능 검증
  - connect_database() 메서드 동작 검증
  - MySQL, PostgreSQL, SQLite, SQL Server 연결 테스트
  - 연결 풀링 및 최적화 동작 검증
  - 보안 연결 및 인증 처리 확인
  - _Requirements: 21.2.4_

- [ ] 2.2.5 대용량 파일 처리 기능 검증
  - load_large_files() 메서드 동작 검증
  - 청킹 및 스트리밍 처리 정확성 테스트
  - 메모리 효율적 처리 성능 검증
  - 진행 상황 추적 시스템 확인
  - _Requirements: 21.2.5_

- [ ] 2.2.6 파일 파싱 오류 처리 기능 검증
  - handle_parsing_errors() 메서드 동작 검증
  - 상세한 오류 메시지 품질 테스트
  - 복구 제안 시스템 적절성 검증
  - 부분 파싱 지원 확인
  - _Requirements: 21.2.6_

- [ ] 2.2.7 데이터 미리보기 기능 검증
  - preview_data() 메서드 동작 검증
  - 샘플 데이터 표시 정확성 테스트
  - 컬럼 정보 및 통계 제공 검증
  - 대화형 미리보기 지원 확인
  - _Requirements: 21.2.7_

- [ ] 2.2.8 데이터 스키마 추론 기능 검증
  - infer_schema() 메서드 동작 검증
  - 자동 컬럼 타입 감지 정확성 테스트
  - 스키마 수정 제안 적절성 검증
  - 스키마 검증 및 최적화 결과 확인
  - _Requirements: 21.2.8_

### 2.3 Data Visualization Agent (포트 8308) 기능 검증

- [ ] 2.3.1 기본 플롯 생성 기능 검증
  - create_basic_plots() 메서드 동작 검증
  - line, bar, scatter, histogram, box plot 생성 테스트
  - 적절한 포맷팅 및 스타일링 검증
  - 자동 축 레이블 및 제목 생성 확인
  - _Requirements: 21.3.1_

- [ ] 2.3.2 고급 플롯 생성 기능 검증
  - create_advanced_plots() 메서드 동작 검증
  - heatmap, violin plot, pair plot, correlation matrix 생성 테스트
  - 복잡한 데이터 관계 시각화 정확성 검증
  - 고급 통계 시각화 지원 확인
  - _Requirements: 21.3.2_

- [ ] 2.3.3 인터랙티브 플롯 생성 기능 검증
  - create_interactive_plots() 메서드 동작 검증
  - Plotly 기반 줌, 호버, 선택 기능 테스트
  - 동적 데이터 탐색 지원 검증
  - 웹 기반 인터랙션 최적화 확인
  - _Requirements: 21.3.3_

- [ ] 2.3.4 통계 플롯 생성 기능 검증
  - create_statistical_plots() 메서드 동작 검증
  - 분포도, Q-Q plot, 회귀 플롯 생성 테스트
  - 통계적 유의성 시각화 정확성 검증
  - 신뢰구간 및 오차 표시 확인
  - _Requirements: 21.3.4_

- [ ] 2.3.5 시계열 플롯 생성 기능 검증
  - create_timeseries_plots() 메서드 동작 검증
  - 시간축 처리 및 최적화 테스트
  - 계절성 분해 시각화 정확성 검증
  - 트렌드 분석 및 예측 표시 확인
  - _Requirements: 21.3.5_

- [ ] 2.3.6 다차원 플롯 생성 기능 검증
  - create_multidimensional_plots() 메서드 동작 검증
  - 3D 플롯 및 서브플롯 생성 테스트
  - 패싯 차트 및 그리드 레이아웃 검증
  - 차원 축소 시각화 지원 확인
  - _Requirements: 21.3.6_

- [ ] 2.3.7 커스텀 스타일링 기능 검증
  - apply_custom_styling() 메서드 동작 검증
  - 테마, 색상, 주석 시스템 테스트
  - 제목, 범례 커스터마이징 검증
  - 브랜딩 및 일관성 유지 확인
  - _Requirements: 21.3.7_

- [ ] 2.3.8 플롯 내보내기 기능 검증
  - export_plots() 메서드 동작 검증
  - PNG, SVG, HTML, PDF 형식 지원 테스트
  - 고해상도 내보내기 옵션 검증
  - 배치 내보내기 지원 확인
  - _Requirements: 21.3.8_

### 2.4 Data Wrangling Agent (포트 8309) 기능 검증

- [ ] 2.4.1 데이터 필터링 기능 검증
  - filter_data() 메서드 동작 검증
  - 복잡한 조건 및 다중 기준 지원 테스트
  - 날짜 범위 필터링 최적화 검증
  - 동적 필터 조건 생성 확인
  - _Requirements: 21.4.1_

- [ ] 2.4.2 데이터 정렬 기능 검증
  - sort_data() 메서드 동작 검증
  - 단일/다중 컬럼 정렬 지원 테스트
  - 커스텀 정렬 순서 구현 검증
  - null 값 처리 옵션 확인
  - _Requirements: 21.4.2_

- [ ] 2.4.3 데이터 그룹화 기능 검증
  - group_data() 메서드 동작 검증
  - 카테고리별 그룹화 최적화 테스트
  - 시간 주기별 그룹화 지원 검증
  - 커스텀 그룹화 함수 구현 확인
  - _Requirements: 21.4.3_

- [ ] 2.4.4 데이터 집계 기능 검증
  - aggregate_data() 메서드 동작 검증
  - sum, mean, count, min, max, percentiles 지원 테스트
  - 커스텀 집계 함수 구현 검증
  - 다중 집계 연산 최적화 확인
  - _Requirements: 21.4.4_

- [ ] 2.4.5 데이터 병합 기능 검증
  - merge_data() 메서드 동작 검증
  - inner, outer, left, right 조인 지원 테스트
  - 키 매칭 최적화 알고리즘 검증
  - 병합 결과 검증 시스템 확인
  - _Requirements: 21.4.5_

- [ ] 2.4.6 데이터 재구성 기능 검증
  - reshape_data() 메서드 동작 검증
  - pivot, melt, transpose 연산 지원 테스트
  - stack, unstack 연산 구현 검증
  - 재구성 결과 최적화 확인
  - _Requirements: 21.4.6_

- [ ] 2.4.7 데이터 샘플링 기능 검증
  - sample_data() 메서드 동작 검증
  - random, stratified, systematic 샘플링 테스트
  - 샘플 크기 최적화 알고리즘 검증
  - 샘플 품질 검증 시스템 확인
  - _Requirements: 21.4.7_

- [ ] 2.4.8 데이터 분할 기능 검증
  - split_data() 메서드 동작 검증
  - train/test/validation 분할 지원 테스트
  - 적절한 비율 자동 계산 검증
  - 분할 결과 검증 및 균형 확인
  - _Requirements: 21.4.8_

### 2.5 Feature Engineering Agent (포트 8310) 기능 검증

- [ ] 2.5.1 수치형 피처 생성 기능 검증
  - create_numerical_features() 메서드 동작 검증
  - polynomial, interaction, ratio, log 피처 생성 테스트
  - 피처 중요도 기반 선택 시스템 검증
  - 수치형 피처 최적화 알고리즘 확인
  - _Requirements: 21.5.1_

- [ ] 2.5.2 범주형 피처 인코딩 기능 검증
  - encode_categorical_features() 메서드 동작 검증
  - one-hot, label, target, binary 인코딩 지원 테스트
  - 고차원 범주형 데이터 최적화 검증
  - 인코딩 품질 검증 시스템 확인
  - _Requirements: 21.5.2_

- [ ] 2.5.3 텍스트 피처 추출 기능 검증
  - extract_text_features() 메서드 동작 검증
  - TF-IDF, word counts, n-grams 지원 테스트
  - 임베딩 기반 피처 추출 검증
  - 텍스트 전처리 최적화 확인
  - _Requirements: 21.5.3_

- [ ] 2.5.4 날짜시간 피처 추출 기능 검증
  - extract_datetime_features() 메서드 동작 검증
  - year, month, day, hour, weekday, season 추출 테스트
  - 시간대 처리 및 정규화 검증
  - 주기성 피처 생성 확인
  - _Requirements: 21.5.4_

- [ ] 2.5.5 피처 스케일링 기능 검증
  - scale_features() 메서드 동작 검증
  - standardization, normalization, robust scaling 테스트
  - 스케일링 방법 자동 선택 검증
  - 스케일링 품질 검증 확인
  - _Requirements: 21.5.5_

- [ ] 2.5.6 피처 선택 기능 검증
  - select_features() 메서드 동작 검증
  - correlation, mutual info, chi-square 기반 선택 테스트
  - recursive elimination 구현 검증
  - 피처 선택 결과 해석 확인
  - _Requirements: 21.5.6_

- [ ] 2.5.7 차원 축소 기능 검증
  - reduce_dimensionality() 메서드 동작 검증
  - PCA, t-SNE, UMAP, factor analysis 지원 테스트
  - 차원 수 자동 최적화 검증
  - 축소 품질 평가 시스템 확인
  - _Requirements: 21.5.7_

- [ ] 2.5.8 피처 중요도 계산 기능 검증
  - calculate_feature_importance() 메서드 동작 검증
  - permutation, SHAP, tree-based importance 테스트
  - 중요도 시각화 및 해석 검증
  - 피처 선택 권장사항 생성 확인
  - _Requirements: 21.5.8_

### 2.6 SQL Database Agent (포트 8311) 기능 검증

- [ ] 2.6.1 데이터베이스 연결 기능 검증
  - connect_database() 메서드 동작 검증
  - 다중 데이터베이스 타입 지원 (MySQL, PostgreSQL, SQLite, SQL Server) 테스트
  - 적절한 인증 및 보안 연결 처리 검증
  - 연결 풀링 및 최적화 시스템 확인
  - _Requirements: 21.6.1_

- [ ] 2.6.2 SQL 쿼리 실행 기능 검증
  - execute_sql_queries() 메서드 동작 검증
  - SELECT, INSERT, UPDATE, DELETE 연산 안전 처리 테스트
  - SQL 인젝션 방지 및 보안 검증 확인
  - 쿼리 결과 포맷팅 및 반환 검증
  - _Requirements: 21.6.2_

- [ ] 2.6.3 복잡한 쿼리 생성 기능 검증
  - create_complex_queries() 메서드 동작 검증
  - JOIN, 서브쿼리, CTE, 윈도우 함수 지원 테스트
  - 쿼리 빌더 패턴 구현 검증
  - 동적 쿼리 생성 시스템 확인
  - _Requirements: 21.6.3_

- [ ] 2.6.4 쿼리 최적화 기능 검증
  - optimize_queries() 메서드 동작 검증
  - 인덱스 제안 시스템 테스트
  - 쿼리 재작성 및 최적화 권장사항 검증
  - 실행 계획 분석 및 해석 확인
  - _Requirements: 21.6.4_

- [ ] 2.6.5 데이터베이스 스키마 분석 기능 검증
  - analyze_database_schema() 메서드 동작 검증
  - 테이블, 컬럼, 관계, 제약조건 분석 테스트
  - 스키마 문서화 및 시각화 검증
  - 데이터 딕셔너리 생성 확인
  - _Requirements: 21.6.5_

- [ ] 2.6.6 데이터 프로파일링 기능 검증
  - profile_database_data() 메서드 동작 검증
  - 분포, 카디널리티, null 비율 분석 테스트
  - 데이터 품질 패턴 감지 검증
  - 통계적 요약 정보 생성 확인
  - _Requirements: 21.6.6_

- [ ] 2.6.7 대용량 쿼리 결과 처리 기능 검증
  - handle_large_query_results() 메서드 동작 검증
  - 페이지네이션 및 스트리밍 처리 테스트
  - 메모리 효율적 결과 처리 검증
  - 진행 상황 추적 시스템 확인
  - _Requirements: 21.6.7_

- [ ] 2.6.8 데이터베이스 오류 처리 기능 검증
  - handle_database_errors() 메서드 동작 검증
  - 의미있는 오류 메시지 제공 테스트
  - 복구 제안 시스템 검증
  - 연결 재시도 및 장애 복구 확인
  - _Requirements: 21.6.8_

### 2.7 EDA Tools Agent (포트 8312) 기능 검증

- [ ] 2.7.1 기술 통계 계산 기능 검증
  - compute_descriptive_statistics() 메서드 동작 검증
  - mean, median, mode, std, skewness, kurtosis 계산 테스트
  - 분포 특성 분석 및 해석 검증
  - 통계적 요약 리포트 생성 확인
  - _Requirements: 21.7.1_

- [ ] 2.7.2 상관관계 분석 기능 검증
  - analyze_correlations() 메서드 동작 검증
  - Pearson, Spearman, Kendall 상관계수 계산 테스트
  - 통계적 유의성 검정 검증
  - 상관관계 시각화 및 해석 확인
  - _Requirements: 21.7.2_

- [ ] 2.7.3 분포 분석 기능 검증
  - analyze_distributions() 메서드 동작 검증
  - 정규성 검정 (Shapiro-Wilk, Kolmogorov-Smirnov) 테스트
  - 분포 적합도 검정 및 Q-Q plot 생성 검증
  - 최적 분포 모델 추천 확인
  - _Requirements: 21.7.3_

- [ ] 2.7.4 범주형 데이터 분석 기능 검증
  - analyze_categorical_data() 메서드 동작 검증
  - 빈도표 생성 및 카이제곱 검정 테스트
  - Cramér's V 계수 계산 검증
  - 범주형 변수 간 연관성 분석 확인
  - _Requirements: 21.7.4_

- [ ] 2.7.5 시계열 분석 기능 검증
  - analyze_time_series() 메서드 동작 검증
  - 트렌드, 계절성, 정상성 검정 테스트
  - 자기상관 분석 (ACF, PACF) 검증
  - 시계열 분해 및 패턴 감지 확인
  - _Requirements: 21.7.5_

- [ ] 2.7.6 이상 감지 기능 검증
  - detect_anomalies() 메서드 동작 검증
  - 이상치, 변화점, 비정상 패턴 감지 테스트
  - 다양한 이상 감지 알고리즘 적용 검증
  - 이상 감지 결과 시각화 및 해석 확인
  - _Requirements: 21.7.6_

- [ ] 2.7.7 데이터 품질 평가 기능 검증
  - assess_data_quality() 메서드 동작 검증
  - 완전성, 일관성, 유효성, 유일성 검사 테스트
  - 데이터 품질 점수 계산 검증
  - 품질 개선 권장사항 제공 확인
  - _Requirements: 21.7.7_

- [ ] 2.7.8 자동화된 인사이트 생성 기능 검증
  - generate_automated_insights() 메서드 동작 검증
  - 주요 발견사항 자동 추출 테스트
  - 서술형 요약 생성 검증
  - 액션 아이템 및 권장사항 도출 확인
  - _Requirements: 21.7.8_

### 2.8 H2O ML Agent (포트 8313) 기능 검증

- [ ] 2.8.1 AutoML 기능 검증
  - run_automl() 메서드 동작 검증
  - 다중 알고리즘 자동 훈련 및 선택 테스트
  - 하이퍼파라미터 자동 최적화 검증
  - 최적 모델 자동 선택 시스템 확인
  - _Requirements: 21.8.1_

- [ ] 2.8.2 분류 모델 훈련 기능 검증
  - train_classification_models() 메서드 동작 검증
  - Random Forest, GBM, XGBoost, Neural Networks 지원 테스트
  - 모델별 최적화 및 튜닝 검증
  - 분류 성능 평가 시스템 확인
  - _Requirements: 21.8.2_

- [ ] 2.8.3 회귀 모델 훈련 기능 검증
  - train_regression_models() 메서드 동작 검증
  - Linear, GLM, GBM, Deep Learning 모델 지원 테스트
  - 회귀 성능 최적화 검증
  - 예측 정확도 평가 시스템 확인
  - _Requirements: 21.8.3_

- [ ] 2.8.4 모델 평가 기능 검증
  - evaluate_models() 메서드 동작 검증
  - accuracy, precision, recall, F1, AUC, RMSE, MAE 계산 테스트
  - 교차 검증 및 성능 안정성 평가 검증
  - 모델 비교 및 선택 지원 확인
  - _Requirements: 21.8.4_

- [ ] 2.8.5 하이퍼파라미터 튜닝 기능 검증
  - tune_hyperparameters() 메서드 동작 검증
  - Grid Search, Random Search, Bayesian Optimization 지원 테스트
  - 효율적 탐색 전략 구현 검증
  - 튜닝 결과 분석 및 시각화 확인
  - _Requirements: 21.8.5_

- [ ] 2.8.6 피처 중요도 분석 기능 검증
  - analyze_feature_importance() 메서드 동작 검증
  - SHAP 값, Permutation Importance, Variable Importance 계산 테스트
  - 피처 기여도 시각화 검증
  - 피처 선택 권장사항 제공 확인
  - _Requirements: 21.8.6_

- [ ] 2.8.7 모델 해석 기능 검증
  - interpret_models() 메서드 동작 검증
  - Partial Dependence Plot, LIME 설명 생성 테스트
  - 모델 의사결정 과정 시각화 검증
  - 비즈니스 친화적 해석 제공 확인
  - _Requirements: 21.8.7_

- [ ] 2.8.8 모델 배포 기능 검증
  - deploy_models() 메서드 동작 검증
  - MOJO, POJO, pickle 형식 모델 내보내기 테스트
  - 모델 버전 관리 시스템 검증
  - 배포 준비 및 검증 프로세스 확인
  - _Requirements: 21.8.8_

### 2.9 MLflow Tools Agent (포트 8314) 기능 검증

- [ ] 2.9.1 실험 추적 기능 검증
  - track_experiments() 메서드 동작 검증
  - 파라미터, 메트릭, 아티팩트, 모델 버전 로깅 테스트
  - 실험 메타데이터 관리 검증
  - 실험 비교 및 분석 지원 확인
  - _Requirements: 21.9.1_

- [ ] 2.9.2 모델 레지스트리 기능 검증
  - manage_model_registry() 메서드 동작 검증
  - 모델 등록, 버전 관리, 스테이지 전환 테스트
  - 모델 승인 워크플로우 검증
  - 모델 계보 추적 시스템 확인
  - _Requirements: 21.9.2_

- [ ] 2.9.3 모델 서빙 기능 검증
  - serve_models() 메서드 동작 검증
  - REST API 엔드포인트 배포 테스트
  - 모델 서빙 최적화 검증
  - 실시간 예측 서비스 구축 확인
  - _Requirements: 21.9.3_

- [ ] 2.9.4 실험 비교 기능 검증
  - compare_experiments() 메서드 동작 검증
  - 런, 메트릭, 파라미터 간 비교 테스트
  - 성능 트렌드 분석 검증
  - 최적 실험 식별 시스템 확인
  - _Requirements: 21.9.4_

- [ ] 2.9.5 아티팩트 관리 기능 검증
  - manage_artifacts() 메서드 동작 검증
  - 데이터셋, 모델, 플롯, 리포트 저장 및 검색 테스트
  - 아티팩트 버전 관리 검증
  - 아티팩트 공유 및 협업 지원 확인
  - _Requirements: 21.9.5_

- [ ] 2.9.6 모델 모니터링 기능 검증
  - monitor_models() 메서드 동작 검증
  - 모델 성능, 드리프트, 데이터 품질 추적 테스트
  - 알림 및 경고 시스템 검증
  - 모니터링 대시보드 구축 확인
  - _Requirements: 21.9.6_

- [ ] 2.9.7 파이프라인 오케스트레이션 기능 검증
  - orchestrate_pipelines() 메서드 동작 검증
  - ML 워크플로우 생성 및 관리 테스트
  - 파이프라인 스케줄링 및 실행 검증
  - 의존성 관리 및 오류 처리 확인
  - _Requirements: 21.9.7_

- [ ] 2.9.8 협업 기능 검증
  - enable_collaboration() 메서드 동작 검증
  - 팀 액세스 권한 관리 테스트
  - 실험 공유 및 권한 설정 검증
  - 협업 워크플로우 지원 확인
  - _Requirements: 21.9.8_

### 2.10 Pandas Analyst Agent (포트 8210) 기능 검증

- [ ] 2.10.1 데이터 로딩 기능 검증
  - load_data_formats() 메서드 동작 검증
  - 다양한 형식 읽기 (CSV, Excel, JSON, Parquet 등) 테스트
  - 적절한 파싱 옵션 자동 설정 검증
  - 로딩 오류 처리 및 복구 확인
  - _Requirements: 21.10.1_

- [ ] 2.10.2 데이터 검사 기능 검증
  - inspect_data() 메서드 동작 검증
  - info, describe, head, tail, shape, dtypes 제공 테스트
  - 데이터 개요 및 요약 통계 검증
  - 데이터 품질 초기 평가 확인
  - _Requirements: 21.10.2_

- [ ] 2.10.3 데이터 선택 기능 검증
  - select_data() 메서드 동작 검증
  - 행 필터링, 컬럼 선택, 데이터 슬라이싱 테스트
  - 복잡한 조건 기반 선택 검증
  - 인덱싱 및 라벨 기반 선택 확인
  - _Requirements: 21.10.3_

- [ ] 2.10.4 데이터 조작 기능 검증
  - manipulate_data() 메서드 동작 검증
  - apply, map, transform, replace, rename 연산 테스트
  - 데이터 변환 및 계산 검증
  - 조건부 데이터 수정 확인
  - _Requirements: 21.10.4_

- [ ] 2.10.5 데이터 집계 기능 검증
  - aggregate_data() 메서드 동작 검증
  - groupby, pivot_table, crosstab, resample 연산 테스트
  - 다차원 집계 및 요약 검증
  - 시간 기반 집계 처리 확인
  - _Requirements: 21.10.5_

- [ ] 2.10.6 데이터 병합 기능 검증
  - merge_data() 메서드 동작 검증
  - merge, join, concat, append 다양한 전략 테스트
  - 키 기반 조인 최적화 검증
  - 병합 결과 검증 및 정리 확인
  - _Requirements: 21.10.6_

- [ ] 2.10.7 데이터 정리 기능 검증
  - clean_data() 메서드 동작 검증
  - 누락값, 중복값, 데이터 타입 처리 테스트
  - 데이터 일관성 확보 검증
  - 정리 과정 문서화 확인
  - _Requirements: 21.10.7_

- [ ] 2.10.8 통계 분석 기능 검증
  - perform_statistical_analysis() 메서드 동작 검증
  - 상관관계, 분포, 가설 검정 테스트
  - 기본 통계 분석 및 해석 검증
  - 통계 결과 시각화 지원 확인
  - _Requirements: 21.10.8_

### 2.11 Report Generator Agent (포트 8316) 기능 검증

- [ ] 2.11.1 경영진 요약 리포트 생성 기능 검증
  - generate_executive_summary() 메서드 동작 검증
  - 고수준 인사이트 및 주요 발견사항 테스트
  - 비즈니스 임팩트 중심 요약 검증
  - 액션 아이템 및 권장사항 확인
  - _Requirements: 21.11.1_

- [ ] 2.11.2 상세 분석 리포트 생성 기능 검증
  - generate_detailed_analysis() 메서드 동작 검증
  - 방법론, 결과, 결론 포함 테스트
  - 기술적 세부사항 및 통계 분석 검증
  - 분석 과정 문서화 확인
  - _Requirements: 21.11.2_

- [ ] 2.11.3 데이터 품질 리포트 생성 기능 검증
  - generate_data_quality_report() 메서드 동작 검증
  - 완전성, 정확성, 일관성 평가 테스트
  - 데이터 품질 메트릭 및 점수 검증
  - 품질 개선 권장사항 확인
  - _Requirements: 21.11.3_

- [ ] 2.11.4 통계 리포트 생성 기능 검증
  - generate_statistical_report() 메서드 동작 검증
  - 기술 통계, 검정, 신뢰구간 테스트
  - 통계적 유의성 해석 검증
  - 통계 결과 시각화 확인
  - _Requirements: 21.11.4_

- [ ] 2.11.5 시각화 리포트 생성 기능 검증
  - generate_visualization_report() 메서드 동작 검증
  - 차트, 테이블, 인터랙티브 요소 포함 테스트
  - 시각화 설명 및 해석 검증
  - 다양한 시각화 형식 지원 확인
  - _Requirements: 21.11.5_

- [ ] 2.11.6 비교 분석 리포트 생성 기능 검증
  - generate_comparative_analysis() 메서드 동작 검증
  - 데이터셋, 시간 주기, 세그먼트 비교 테스트
  - 변화 추이 및 패턴 분석 검증
  - 비교 결과 시각화 확인
  - _Requirements: 21.11.6_

- [ ] 2.11.7 권장사항 리포트 생성 기능 검증
  - generate_recommendation_report() 메서드 동작 검증
  - 실행 가능한 인사이트 제공 테스트
  - 다음 단계 및 액션 플랜 검증
  - 우선순위 및 임팩트 평가 확인
  - _Requirements: 21.11.7_

- [ ] 2.11.8 리포트 내보내기 기능 검증
  - export_reports() 메서드 동작 검증
  - PDF, HTML, Word, PowerPoint 형식 지원 테스트
  - 템플릿 기반 리포트 생성 검증
  - 브랜딩 및 스타일 커스터마이징 확인
  - _Requirements: 21.11.8_

## Phase 3: LLM First 핵심 시스템 구조 구현

### 1.1 SmartQueryRouter 완전 구현

- [ ] 1.1.1 SmartQueryRouter 클래스 기본 구조 생성
  - SmartQueryRouter 클래스 정의 및 초기화 메서드 구현
  - LLMFactory, AgentPool, LangfuseIntegration 의존성 주입
  - 기본 설정 및 상태 관리 시스템 구현
  - _Requirements: 6.5, 6.6, 6.7_

- [ ] 1.1.2 빠른 복잡도 사전 판단 시스템 구현
  - quick_complexity_assessment() 메서드 구현
  - LLM 기반 복잡도 판단 프롬프트 최적화 (trivial|simple|medium|complex)
  - 판단 기준 및 신뢰도 평가 시스템 구현
  - JSON 응답 파싱 및 검증 로직 구현
  - _Requirements: 6.2_

- [ ] 1.1.3 Direct Response 시스템 구현 (5-10초)
  - direct_response() 메서드 구현
  - Langfuse 간단 세션 추적 시스템 구현
  - 직접 LLM 응답 생성 및 스트리밍 처리
  - 오류 처리 및 복구 메커니즘 구현
  - _Requirements: 6.5_

- [ ] 1.1.4 Single Agent Response 시스템 구현 (10-20초)
  - single_agent_response() 메서드 구현
  - select_best_single_agent() LLM 기반 에이전트 선택 로직
  - 단일 에이전트 실행 및 결과 처리 시스템
  - 에이전트별 특화 처리 로직 구현
  - _Requirements: 6.6_

- [ ] 1.1.5 Multi-Agent Orchestration 연동 구현 (30-60초)
  - orchestrated_response() 메서드 구현
  - LLMFirstOptimizedOrchestrator와의 완전한 연동
  - 복잡한 쿼리 처리 워크플로우 구현
  - 진행 상황 추적 및 사용자 피드백 시스템
  - _Requirements: 6.7_

### 1.2 LLMFirstOptimizedOrchestrator 핵심 구현

- [ ] 1.2.1 LLMFirstOptimizedOrchestrator 클래스 기본 구조
  - 클래스 정의 및 의존성 주입 시스템 구현
  - LLMFactory, AgentPool, StreamingManager, LangfuseIntegration 통합
  - SeparatedCritiqueSystem, SeparatedReplanningSystem 연동
  - 기본 설정 및 상태 관리 시스템 구현
  - _Requirements: 6.1, 6.4_

- [ ] 1.2.2 LLM 기반 통합 복잡도 분석 및 전략 결정 시스템
  - analyze_and_strategize_llm_first() 메서드 구현
  - 통합 프롬프트 설계 (복잡도 분석 + 전략 결정 + 에이전트 계획)
  - JSON 응답 구조 정의 및 파싱 시스템 구현
  - qwen3-4b-fast 모델 특성 고려한 최적화 로직
  - _Requirements: 6.2, 6.4_

- [ ] 1.2.3 적응형 실행 전략 구현
  - execute_fast_track() 메서드 구현 (단순 쿼리용)
  - execute_balanced() 메서드 구현 (일반 쿼리용)
  - execute_thorough() 메서드 구현 (복잡 쿼리용)
  - execute_expert_mode() 메서드 구현 (전문가 쿼리용)
  - 각 모드별 최적화된 처리 로직 구현
  - _Requirements: 6.4_

- [ ] 1.2.4 에이전트 실행 및 스트리밍 시스템
  - execute_agents_streaming() 메서드 구현
  - 병렬/순차 에이전트 실행 로직 구현
  - 실시간 진행 상황 스트리밍 시스템
  - 0.001초 지연 최적화 구현
  - _Requirements: 8.1, 8.2, 8.3_

### 1.3 분리된 Critique & Replanning 시스템 구현

- [ ] 1.3.1 SeparatedCritiqueSystem 구현
  - SeparatedCritiqueSystem 클래스 정의 및 초기화
  - perform_separated_critique() 메서드 구현
  - 순수 평가 역할 프롬프트 설계 (해결책 제안 금지)
  - 4가지 평가 기준 구현 (정확성, 완전성, 품질, 일관성)
  - JSON 응답 구조 및 파싱 시스템 구현
  - _Requirements: 6.3_

- [ ] 1.3.2 SeparatedReplanningSystem 구현
  - SeparatedReplanningSystem 클래스 정의 및 초기화
  - perform_separated_replanning() 메서드 구현
  - 순수 재계획 역할 프롬프트 설계 (평가 없음)
  - 개선된 계획 수립 로직 구현
  - 리소스 조정 및 예상 개선점 계산 시스템
  - _Requirements: 6.3_

- [ ] 1.3.3 LLMFirstComplexityAnalyzer 구현
  - LLMFirstComplexityAnalyzer 클래스 구현
  - analyze_query_complexity_llm_first() 메서드 구현
  - 5차원 복잡도 분석 시스템 (구조적, 도메인, 의도, 데이터, 협업)
  - 하드코딩 규칙 제거 및 순수 LLM 기반 판단
  - 처리 전략 및 리스크 요소 분석 시스템
  - _Requirements: 6.2_

## Phase 2: A2A SDK 0.2.9 완전 준수 구현

### 2.1 A2A SDK 0.2.9 Import 패턴 업데이트

- [ ] 2.1.1 모든 에이전트 Import 패턴 업데이트
  - 11개 에이전트 파일에서 A2AStarletteApplication import 적용
  - 기존 deprecated import 패턴 제거
  - 새로운 import 구조 검증 및 테스트
  - _Requirements: 12.1_

- [ ] 2.1.2 DefaultRequestHandler 패턴 적용
  - 모든 에이전트에 DefaultRequestHandler import 및 초기화
  - 올바른 초기화 패턴 구현
  - 요청 처리 로직 업데이트
  - _Requirements: 12.2_

- [ ] 2.1.3 AgentExecutor 및 RequestContext 업데이트
  - AgentExecutor, RequestContext import 및 사용법 업데이트
  - 올바른 async 패턴 구현
  - 컨텍스트 관리 시스템 구현
  - _Requirements: 12.3_

- [ ] 2.1.4 AgentCapabilities 설정 구현
  - 모든 에이전트에 AgentCapabilities(streaming=True) 설정
  - 스트리밍 기능 활성화 및 검증
  - 에이전트 카드 정보 업데이트
  - _Requirements: 12.4_

### 2.2 에이전트 통신 및 발견 시스템

- [ ] 2.2.1 A2A 프로토콜 기반 통신 시스템
  - 에이전트 간 메시지 패싱 시스템 구현
  - A2A 프로토콜 사양 완전 준수
  - 통신 오류 처리 및 재시도 로직
  - _Requirements: 3.2_

- [ ] 2.2.2 동적 에이전트 발견 시스템
  - 에이전트 자동 발견 및 등록 시스템
  - 에이전트 카드 정보 수집 및 관리
  - 실시간 에이전트 상태 모니터링
  - _Requirements: 3.3_

- [ ] 2.2.3 에이전트 장애 복구 시스템
  - 에이전트 장애 감지 및 알림 시스템
  - 자동 failover 및 복구 메커니즘
  - 우아한 성능 저하 처리 로직
  - _Requirements: 18.1, 18.2_

## Phase 3: Langfuse v2 통합 및 SSE 스트리밍 최적화

### 3.1 Langfuse v2.60.8 세션 기반 추적 시스템 구현

- [ ] 3.1.1 SessionBasedTracer 구현
  - SessionBasedTracer 클래스 구현
  - session ID 형식 `user_query_{timestamp}_{user_id}` 적용
  - EMP_NO=2055186을 user_id로 설정
  - 세션 메타데이터 관리 시스템 구현
  - _Requirements: 13.1_

- [ ] 3.1.2 LangfuseEnhancedA2AExecutor 구현
  - 에이전트 실행 자동 추적 래퍼 구현
  - A2A 에이전트 호출 시 자동 trace 생성
  - 에이전트 성능 메트릭 수집 및 로깅
  - 멀티에이전트 워크플로우 추적 시스템
  - _Requirements: 13.2_

- [ ] 3.1.3 RealTimeStreamingTaskUpdater 구현
  - 스트리밍 중 trace 컨텍스트 유지 시스템
  - SSE 스트리밍과 Langfuse 추적 통합
  - 실시간 진행 상황 추적 및 로깅
  - 스트리밍 완료 시 종합 메타데이터 생성
  - _Requirements: 13.3, 13.4_

### 3.2 SSE 스트리밍 최적화 구현

- [ ] 3.2.1 process_query_streaming() 최적화
  - async def process_query_streaming() 메서드 구현
  - 적절한 SSE 헤더 설정 및 CORS 처리
  - 청크 기반 스트리밍 응답 최적화
  - 스트리밍 오류 처리 및 복구 메커니즘
  - _Requirements: 14.1_

- [ ] 3.2.2 0.001초 지연 최적화 구현
  - await asyncio.sleep(0.001) 적용으로 부드러운 UX 구현
  - 청크 크기 최적화 및 버퍼링 전략
  - 네트워크 지연 고려한 적응형 지연 조정
  - 사용자 경험 최적화를 위한 진행 표시
  - _Requirements: 14.2_

- [ ] 3.2.3 멀티에이전트 협업 스트리밍 구현
  - async for chunk_data in client.stream_task() 구현
  - 에이전트 간 협업 과정 실시간 표시
  - 각 에이전트 작업 상태 및 진행률 스트리밍
  - 최종 결과 누적 및 완전한 분석 결과 제공
  - _Requirements: 14.3, 14.4_

## Phase 4: Playwright MCP E2E 테스트 시스템 구현

### 4.1 Playwright MCP 통합 테스트 구현

- [ ] 4.1.1 Playwright 기본 테스트 환경 구축
  - from playwright.async_api import async_playwright 설정
  - 브라우저 자동화 환경 구성 및 최적화
  - 테스트 데이터 준비 및 관리 시스템
  - 스크린샷 및 테스트 결과 저장 시스템
  - _Requirements: 15.1_

- [ ] 4.1.2 Streamlit UI 상호작용 테스트
  - await page.wait_for_selector('[data-testid="stApp"]') 구현
  - Streamlit 컴포넌트 자동 감지 및 상호작용
  - 파일 업로드 시뮬레이션 및 검증
  - UI 응답성 및 사용자 경험 테스트
  - _Requirements: 15.2, 15.3_

- [ ] 4.1.3 멀티에이전트 협업 검증 테스트
  - 실제 CSV 파일 업로드 및 처리 검증
  - 에이전트 간 협업 과정 스크린샷 캡처
  - 분석 결과 정확성 및 완전성 검증
  - E2E 워크플로우 성능 측정 및 최적화
  - _Requirements: 15.4_

### 4.2 도메인 전문성 테스트 구현

- [ ] 4.2.1 일반 사용자 시나리오 테스트
  - 간단한 데이터 업로드 및 기본 분석 시나리오
  - 직관적 UI/UX 사용성 검증
  - 초보자도 쉽게 사용할 수 있는지 확인
  - 오류 상황에서의 사용자 가이드 검증
  - _Requirements: 22.1_

- [ ] 4.2.2 전문가 시나리오 테스트 (ion_implant_3lot_dataset.csv)
  - ion_implant_3lot_dataset.csv 업로드 및 처리
  - query.txt 내용 기반 복잡한 도메인 분석
  - 반도체 도메인 특성 자동 감지 검증
  - LLM 기반 동적 전문 지식 적응 확인
  - _Requirements: 22.2, 19.1, 19.2, 19.3, 19.4_

## Phase 5: 시스템 통합 및 최적화

### 5.1 ChatGPT 스타일 UI/UX 완성

- [ ] 5.1.1 ChatGPT 스타일 채팅 인터페이스 구현
  - 대화형 채팅 UI 구현 및 최적화
  - 드래그앤드롭 파일 업로드 영역 구현
  - 실시간 분석 진행 상황 시각화
  - 에이전트 협업 과정 사용자 친화적 표시
  - _Requirements: 20.1, 20.3_

- [ ] 5.1.2 파일 업로드 및 처리 시스템 최적화
  - CSV/Excel 파일 드래그앤드롭 지원
  - 파일 업로드 시 시각적 피드백 제공
  - 대용량 파일 처리 진행률 표시
  - 파일 형식 검증 및 오류 처리
  - _Requirements: 20.2_

- [ ] 5.1.3 분석 결과 종합 리포트 시스템
  - 코드, 차트, 인사이트 포함한 종합 리포트
  - ChatGPT Data Analyst 수준의 결과 표시
  - 인터랙티브 차트 및 시각화 지원
  - 결과 내보내기 및 공유 기능
  - _Requirements: 20.4, 9.1, 9.2_

### 5.2 qwen3-4b-fast 모델 최적화

- [ ] 5.2.1 LLM Factory 통합 및 최적화
  - create_llm_instance() 자동 Langfuse 콜백 주입
  - OLLAMA_BASE_URL=http://localhost:11434 설정
  - qwen3-4b-fast 모델 특성에 맞는 파라미터 튜닝
  - 로컬 환경 제약 고려한 성능 최적화
  - _Requirements: 17.1, 17.2, 17.3, 17.4_

- [ ] 5.2.2 성능 및 품질 균형 최적화
  - 데이터 분석용 temperature 및 token limit 설정
  - 수치 결과 정확성 보장 및 환각 방지
  - 신뢰할 수 있는 데이터 기반 인사이트 생성
  - 로컬 처리 속도와 분석 품질 균형 달성
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

### 5.3 MCP 서버 통합 및 하이브리드 아키텍처

- [ ] 5.3.1 MCPConfigManager 구현
  - MCP 서버 포트 설정 (8006-8020) 관리
  - MCP 서버 자동 시작 및 상태 모니터링
  - A2A 에이전트와 MCP 도구 간 협업 시스템
  - 하이브리드 워크플로우 오케스트레이션
  - _Requirements: 16.1, 16.3_

- [ ] 5.3.2 도메인 적응형 MCP 도구 통합
  - LLM 추론을 통한 도메인 적응 MCP 도구 발견
  - 범용 도메인 적응 도구 구현 및 통합
  - MCP 서버 health check 및 복구 시스템
  - A2A+MCP 하이브리드 아키텍처 최적화
  - _Requirements: 16.2, 16.4_



