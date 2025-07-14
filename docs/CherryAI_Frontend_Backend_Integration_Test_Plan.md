# 🍒 CherryAI 프론트엔드-백엔드 완전 융합 테스트 계획

## 📋 개요

CherryAI의 ChatGPT/Claude 수준 UI/UX 구현과 백엔드 기능 완전 융합을 위한 체계적인 테스트 계획입니다.

### 🎯 핵심 원칙
- **LLM First**: Rule 기반 하드코딩/패턴 매칭 절대 금지
- **A2A SDK 0.2.9**: 표준 완전 준수
- **SSE 스트리밍**: WebSocket 사용 금지, 순수 SSE만 사용
- **근본적 해결**: 현상 해결이 아닌 원인 분석 및 개선

---

## 🔧 테스트 아키텍처

### 1. 단위 테스트 (pytest)
```
tests/unit/
├── ui/
│   ├── test_chat_interface.py
│   ├── test_rich_content_renderer.py
│   ├── test_streaming_manager.py
│   └── test_session_manager.py
├── core/
│   ├── test_frontend_backend_bridge.py
│   ├── test_llm_first_integration.py
│   └── test_knowledge_bank_integration.py
└── integration/
    ├── test_a2a_agents_integration.py
    ├── test_mcp_tools_integration.py
    └── test_sse_streaming.py
```

### 2. 통합 테스트 (pytest)
```
tests/integration/
├── test_complete_workflow.py
├── test_multi_agent_collaboration.py
├── test_file_processing_pipeline.py
└── test_session_persistence.py
```

### 3. E2E UI 테스트 (Playwright MCP)
```
tests/e2e/
├── test_chat_interface_e2e.py
├── test_file_upload_e2e.py
├── test_multi_agent_workflow_e2e.py
└── test_session_management_e2e.py
```

---

## 📊 테스트 시나리오

### 🎯 Scenario 1: 기본 채팅 인터페이스
**목표**: ChatGPT/Claude 수준의 대화 경험 검증

#### A. 단위 테스트
- [ ] Enter키 실행 테스트
- [ ] Shift+Enter 멀티라인 입력 테스트
- [ ] 메시지 히스토리 저장/불러오기
- [ ] 타이핑 효과 렌더링
- [ ] 에러 메시지 표시

#### B. 통합 테스트
- [ ] LLM First Engine 연동
- [ ] Knowledge Bank 검색 연동
- [ ] SSE 스트리밍 연동

#### C. E2E 테스트 (Playwright)
```
1. 브라우저에서 CherryAI 접속
2. 채팅창에 "안녕하세요" 입력
3. Enter키로 전송
4. AI 응답 실시간 스트리밍 확인
5. Shift+Enter로 멀티라인 입력 테스트
6. 메시지 히스토리 표시 확인
```

### 🎯 Scenario 2: 파일 업로드 및 데이터 분석
**목표**: 전체 데이터 분석 워크플로우 검증

#### A. 단위 테스트
- [ ] 파일 업로드 프로세서
- [ ] pandas 지원 파일 형식 검증 (CSV, Excel, JSON)
- [ ] 파일 메타데이터 추출
- [ ] Knowledge Bank 저장

#### B. 통합 테스트
- [ ] 파일 → 지식 추출 → 에이전트 전달
- [ ] 에이전트 선택 로직 (LLM 기반)
- [ ] 다중 에이전트 협업

#### C. E2E 테스트 (Playwright)
```
1. 테스트 CSV 파일 준비 (iris.csv)
2. 드래그앤드롭으로 파일 업로드
3. "이 데이터의 패턴을 분석해줘" 입력
4. LLM First Engine의 의도 분석 확인
5. 적절한 A2A 에이전트 선택 확인
6. 실시간 분석 진행 상황 표시
7. 차트/표/통계 결과 렌더링 검증
8. 다운로드 기능 테스트
```

### 🎯 Scenario 3: 멀티 에이전트 협업
**목표**: 11개 A2A 에이전트의 협업 검증

#### A. A2A 에이전트 개별 테스트
- [ ] **data_cleaning**: 결측값, 이상치 처리
- [ ] **data_loader**: 다양한 형식 로드
- [ ] **data_visualization**: 차트 생성
- [ ] **data_wrangling**: 데이터 변환
- [ ] **eda_tools**: 탐색적 데이터 분석
- [ ] **feature_engineering**: 특성 생성
- [ ] **h2o_ml**: AutoML 모델링
- [ ] **mlflow_tools**: 실험 추적
- [ ] **sql_database**: DB 연동
- [ ] **pandas_collaboration_hub**: 중앙 조정
- [ ] **orchestrator**: 전체 오케스트레이션

#### B. 에이전트 간 협업 테스트
- [ ] 순차적 협업 (데이터 로드 → 정제 → 분석)
- [ ] 병렬 협업 (여러 차트 동시 생성)
- [ ] 조건부 협업 (데이터 품질에 따른 분기)

#### C. E2E 테스트 (Playwright)
```
1. 복잡한 데이터셋 업로드 (결측값, 다양한 타입 포함)
2. "완전한 데이터 분석 보고서를 만들어줘" 요청
3. 여러 에이전트 작업 상태 실시간 확인
4. 에이전트 간 협업 플로우 시각화 검증
5. 최종 결과물의 완성도 평가
```

### 🎯 Scenario 4: MCP 도구 통합
**목표**: 7개 MCP 도구의 완전한 통합 검증

#### A. MCP 도구 개별 테스트
- [ ] **playwright**: UI 자동화
- [ ] **file_manager**: 파일 관리
- [ ] **database_connector**: DB 연결
- [ ] **api_gateway**: API 호출
- [ ] **data_analyzer**: 고급 분석
- [ ] **chart_generator**: 차트 생성
- [ ] **llm_gateway**: LLM 연동

#### B. A2A-MCP 통합 테스트
- [ ] A2A 에이전트가 MCP 도구 호출
- [ ] MCP 도구 결과를 A2A로 전달
- [ ] 에러 처리 및 복구

#### C. E2E 테스트 (Playwright)
```
1. "웹사이트에서 데이터를 스크래핑해서 분석해줘" 요청
2. Playwright MCP를 통한 웹 스크래핑
3. 스크래핑 데이터를 A2A 에이전트로 전달
4. 분석 결과를 Chart Generator로 시각화
5. 최종 리포트를 File Manager로 저장
```

### 🎯 Scenario 5: 세션 관리
**목표**: 세션 생성, 저장, 불러오기, 삭제 기능 검증

#### A. 단위 테스트
- [ ] 세션 생성 및 ID 할당
- [ ] 세션 메타데이터 저장
- [ ] 세션 목록 조회
- [ ] 세션 삭제 기능
- [ ] 세션 검색 및 필터링

#### B. 통합 테스트
- [ ] Knowledge Bank와 세션 연동
- [ ] 세션 간 데이터 격리
- [ ] 세션 복구 메커니즘

#### C. E2E 테스트 (Playwright)
```
1. 새 세션에서 대화 시작
2. 파일 업로드 및 분석 수행
3. 세션 목록에서 현재 세션 확인
4. 다른 세션으로 전환
5. 이전 세션으로 돌아가서 히스토리 확인
6. 세션 삭제 기능 테스트
7. 삭제된 세션이 목록에서 제거되는지 확인
```

### 🎯 Scenario 6: SSE 스트리밍 성능
**목표**: 실시간 스트리밍의 안정성과 성능 검증

#### A. 단위 테스트
- [ ] SSE 연결 설정
- [ ] 청크 단위 전송
- [ ] 연결 끊김 처리
- [ ] 에러 복구

#### B. 통합 테스트
- [ ] 대용량 응답 스트리밍
- [ ] 다중 사용자 동시 스트리밍
- [ ] 네트워크 지연 시뮬레이션

#### C. E2E 테스트 (Playwright)
```
1. 대용량 데이터 분석 요청
2. 실시간 스트리밍 응답 확인
3. 스트리밍 중 페이지 새로고침
4. 연결 복구 확인
5. 스트리밍 완료 후 최종 결과 검증
```

### 🎯 Scenario 7: 리치 콘텐츠 렌더링
**목표**: 차트, 코드, 테이블 등의 완벽한 렌더링 검증

#### A. 단위 테스트
- [ ] Plotly 차트 렌더링
- [ ] 코드 블록 하이라이팅
- [ ] 데이터프레임 테이블 표시
- [ ] 이미지 표시 및 다운로드

#### B. 통합 테스트
- [ ] 에이전트 결과물 → 렌더링 파이프라인
- [ ] 인터랙티브 요소 동작
- [ ] 반응형 레이아웃

#### C. E2E 테스트 (Playwright)
```
1. "데이터를 다양한 차트로 시각화해줘" 요청
2. 생성된 차트들의 인터랙티브 기능 테스트
3. 차트 확대/축소, 범례 클릭 등
4. 코드 블록의 복사 버튼 테스트
5. 테이블의 정렬/필터 기능 테스트
```

---

## 🚨 오류 처리 및 복구 테스트

### Critical Path Testing
- [ ] A2A 서버 다운 시 복구
- [ ] MCP 도구 연결 실패 처리
- [ ] OpenAI API 한도 초과 시 fallback
- [ ] Knowledge Bank 연결 실패 처리
- [ ] 세션 데이터 손실 복구

### Performance Testing
- [ ] 동시 사용자 100명 시뮬레이션
- [ ] 대용량 파일 (100MB+) 처리
- [ ] 장시간 세션 안정성
- [ ] 메모리 누수 검사

---

## 📈 성공 기준

### 기능적 요구사항
- ✅ 모든 A2A 에이전트 정상 동작 (11개)
- ✅ 모든 MCP 도구 정상 동작 (7개)
- ✅ SSE 스트리밍 100% 안정성
- ✅ 세션 관리 완전 기능
- ✅ LLM First 원칙 완전 준수

### 성능 요구사항
- ✅ 응답 시작 시간 < 2초
- ✅ 스트리밍 지연 < 100ms
- ✅ 파일 업로드 처리 < 10초
- ✅ 세션 전환 < 1초

### 사용성 요구사항
- ✅ ChatGPT/Claude 수준의 UX
- ✅ 에러 메시지 명확성
- ✅ 접근성 (스크린 리더 지원)
- ✅ 반응형 디자인

---

## 🔄 테스트 실행 순서

1. **Phase 1**: 단위 테스트 (pytest)
2. **Phase 2**: 통합 테스트 (pytest) 
3. **Phase 3**: E2E 테스트 (Playwright MCP)
4. **Phase 4**: 성능 테스트
5. **Phase 5**: 오류 처리 테스트
6. **Phase 6**: 최종 검증

각 Phase에서 실패 시 근본 원인 분석 후 다음 단계 진행.

---

## 📝 테스트 리포트

각 테스트 완료 후 다음 정보를 기록:
- 테스트 결과 (통과/실패)
- 실행 시간
- 발견된 이슈
- 근본 원인 분석
- 개선 방안
- LLM First 원칙 준수 여부 