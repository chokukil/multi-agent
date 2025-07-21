# CherryAI 통합 시스템 구현 작업 목록

## 📋 개요

이 문서는 **검증 완료된 LLM-First Universal Engine 백엔드**와 **ChatGPT 스타일 UI/UX**를 통합하는 구체적인 작업 목록입니다.

### 통합 전략
- **✅ 백엔드 보존**: 검증된 Universal Engine 구조 100% 유지 (수정 금지)
- **🆕 UI 레이어 추가**: 새로운 cherry_ai.py와 통합 컴포넌트 구현
- **🔗 브리지 패턴**: 기존 백엔드와 새 UI 간 브리지 레이어 구축
- **📈 기능 확장**: Langfuse v2, SSE 스트리밍, E2E 테스트 추가

## 🎯 구현 작업 목록

### Phase 1: 핵심 통합 브리지 구현 (3-5일)

- [ ] 1. Universal Engine Bridge 구현
  - 검증된 UniversalQueryProcessor와 새 UI 간 연결 브리지 생성
  - 기존 MetaReasoningEngine, DynamicContextDiscovery 등 모든 컴포넌트 활용
  - A2AAgentDiscoverySystem과 A2AWorkflowOrchestrator 통합 유지
  - 스트리밍 레이어 추가 (기존 로직은 수정 없음)
  - _Requirements: 1.1, 2.1_
  - _구현 위치: integration/universal_engine_bridge.py_

- [ ] 1.1 ChatGPT 스타일 메인 인터페이스 구현
  - Streamlit 기반 ChatGPT 스타일 채팅 인터페이스 구현
  - 사용자 입력, 메시지 히스토리, 응답 표시 기능
  - 파일 업로드 드래그 앤 드롭 인터페이스
  - 실시간 에이전트 상태 표시 사이드바
  - _Requirements: 2.1, 4.1_
  - _구현 위치: ui/chatgpt_interface.py_

- [ ] 1.2 SSE 스트리밍 핸들러 구현
  - Server-Sent Events 기반 실시간 스트리밍 시스템
  - 0.001초 지연 간격으로 부드러운 청크 단위 전송
  - Universal Engine 처리 과정의 실시간 표시
  - 에이전트 협업 과정 스트리밍 시각화
  - _Requirements: 5.1, 5.2_
  - _구현 위치: ui/streaming_handler.py_

- [ ] 1.3 A2A 에이전트 연결 관리자 구현
  - 기존 12개 A2A 에이전트 (포트 8306-8316) 연결 관리
  - 실시간 에이전트 상태 모니터링 및 헬스체크
  - 에이전트 장애 감지 및 복구 메커니즘
  - start.sh/stop.sh와 연동된 에이전트 생명주기 관리
  - _Requirements: 3.1, 9.1_
  - _구현 위치: integration/a2a_agent_connector.py_

### Phase 2: Langfuse v2 통합 및 세션 관리 (2-3일)

- [ ] 2. Langfuse v2 세션 추적 시스템 구현
  - Langfuse v2 SDK를 활용한 세션 기반 추적 시스템
  - EMP_NO=2055186을 user_id로 사용하는 세션 관리
  - `user_query_{timestamp}_{user_id}` 형식의 세션 ID 생성
  - Universal Engine 컴포넌트별 상세 추적 로깅
  - _Requirements: 4.1, 4.2_
  - _구현 위치: integration/langfuse_session_tracer.py_

- [ ] 2.1 멀티 에이전트 협업 추적 구현
  - 단일 세션 내 모든 A2A 에이전트 활동 통합 추적
  - 에이전트별 실행 시간, 입출력 데이터, 성능 메트릭 기록
  - 워크플로우 의존성 및 협업 과정 시각화 데이터 생성
  - 세션 완료 시 종합 성능 리포트 생성
  - _Requirements: 4.3, 4.4_
  - _구현 위치: integration/langfuse_session_tracer.py (확장)_

- [ ] 2.2 실시간 스트리밍 태스크 업데이터 구현
  - 스트리밍 중 Langfuse 추적 컨텍스트 유지
  - RealTimeStreamingTaskUpdater 클래스 구현
  - 청크 단위 스트리밍과 추적 데이터 동기화
  - 스트리밍 성능 메트릭 및 사용자 경험 지표 수집
  - _Requirements: 5.3, 5.4_
  - _구현 위치: integration/streaming_task_updater.py_

### Phase 3: 전문가 시나리오 및 추천 시스템 (2-3일)

- [ ] 3. 전문가 시나리오 핸들러 구현
  - ion_implant_3lot_dataset.csv 자동 감지 및 로드
  - query.txt 내용을 1회성으로 읽어 도메인 지식 통합
  - 반도체 도메인 컨텍스트 자동 감지 (Universal Engine 활용)
  - 20년 경력 엔지니어 수준의 전문 분석 결과 생성
  - _Requirements: 6.1, 6.2, 6.3_
  - _구현 위치: integration/expert_scenario_handler.py_

- [ ] 3.1 지능형 분석 추천 엔진 구현
  - 데이터 업로드 시 최대 3개 초기 분석 추천 생성
  - Universal Engine의 DynamicContextDiscovery 활용한 데이터 특성 분석
  - 분석 완료 후 LLM 기반 후속 분석 제안
  - 클릭 한 번으로 실행 가능한 추천 버튼 인터페이스
  - _Requirements: 7.1, 7.2, 10.1_
  - _구현 위치: services/recommendation_service.py_

- [ ] 3.2 결과 렌더링 시스템 구현
  - 코드, 차트, 테이블 통합 렌더링 시스템
  - Plotly 차트 인터랙티브 시각화 지원
  - 코드 하이라이팅 및 실행 결과 연결 표시
  - 다양한 형식 (CSV, PNG, HTML) 내보내기 기능
  - _Requirements: 2.4, 10.4_
  - _구현 위치: ui/result_renderer.py_

### Phase 4: 시스템 관리 및 성능 최적화 (2-3일)

- [ ] 4. 시스템 관리 스크립트 개선
  - start.sh 스크립트 개선: 12개 A2A 에이전트 + 오케스트레이터 순차 시작
  - stop.sh 스크립트 개선: 우아한 종료 및 리소스 정리
  - 에이전트 상태 모니터링 및 자동 재시작 메커니즘
  - 시스템 헬스체크 및 진단 도구 통합
  - _Requirements: 9.1, 9.2, 9.3_
  - _구현 위치: start.sh, stop.sh (개선)_

- [ ] 4.1 성능 모니터링 서비스 구현
  - 검증된 45초 평균 응답 시간 유지 모니터링
  - qwen3-4b-fast 모델 성능 최적화 및 추적
  - 메모리 사용량, CPU 사용률 실시간 모니터링
  - 성능 저하 감지 시 자동 최적화 트리거
  - _Requirements: 8.4, 8.5_
  - _구현 위치: services/performance_service.py_

- [ ] 4.2 에러 처리 및 복구 시스템 구현
  - Universal Engine 기반 지능형 에러 처리
  - 에이전트 장애 시 자동 대체 에이전트 선택
  - 사용자 친화적 에러 메시지 및 복구 제안
  - 시스템 복구 및 상태 복원 메커니즘
  - _Requirements: 9.4, 12.3_
  - _구현 위치: services/error_handling_service.py_

### Phase 5: E2E 테스트 및 검증 시스템 (2-3일)

- [ ] 5. Playwright MCP 통합 E2E 테스트 구현
  - Playwright MCP를 활용한 브라우저 자동화 테스트
  - 일반 사용자 시나리오: 기본 데이터 분석 워크플로우 테스트
  - 전문가 시나리오: ion_implant_3lot_dataset.csv + query.txt 완전 테스트
  - 멀티 에이전트 협업 과정 자동 검증
  - _Requirements: 7.1, 7.2, 7.3_
  - _구현 위치: tests/e2e/playwright_mcp_tests.py_

- [ ] 5.1 스트리밍 기능 E2E 테스트 구현
  - SSE 스트리밍 기능 자동 테스트
  - 0.001초 지연 간격 정확성 검증
  - 실시간 업데이트 및 사용자 경험 테스트
  - 스트리밍 중 에러 처리 및 복구 테스트
  - _Requirements: 7.4, 7.5_
  - _구현 위치: tests/e2e/streaming_e2e_tests.py_

- [ ] 5.2 통합 시스템 회귀 테스트 구현
  - 검증된 Universal Engine 기능 무결성 확인
  - UI 통합 후 기존 성능 지표 유지 검증
  - 모든 26개 컴포넌트 및 19개 메서드 정상 동작 확인
  - Zero-hardcoding 아키텍처 준수 지속 검증
  - _Requirements: 13.1, 13.2, 13.3_
  - _구현 위치: tests/integration/regression_tests.py_

### Phase 6: 최종 통합 및 배포 준비 (1-2일)

- [ ] 6. 메인 cherry_ai.py 통합 구현
  - 모든 컴포넌트를 통합하는 메인 애플리케이션 파일
  - Streamlit 앱 설정 및 라우팅 구현
  - 세션 상태 관리 및 사용자 컨텍스트 유지
  - 전체 시스템 초기화 및 종료 로직
  - _Requirements: 모든 요구사항 통합_
  - _구현 위치: cherry_ai.py (새 파일)_

- [ ] 6.1 설정 및 환경 관리 시스템 구현
  - UI, 스트리밍, Langfuse 설정 통합 관리
  - 환경 변수 기반 동적 설정 로드
  - 개발/프로덕션 환경별 설정 분리
  - 설정 변경 시 실시간 반영 메커니즘
  - _Requirements: 8.1, 8.2_
  - _구현 위치: config/ 디렉토리 전체_

- [ ] 6.2 최종 통합 테스트 및 검증
  - 전체 시스템 통합 테스트 실행
  - 성능 벤치마크 및 품질 지표 검증
  - 사용자 시나리오 기반 종합 테스트
  - 프로덕션 배포 준비 상태 확인
  - _Requirements: 13.4, 13.5_
  - _구현 위치: tests/final_integration_test.py_

## 📊 구현 우선순위 및 일정

### 🚀 Critical Path (핵심 경로)
1. **Phase 1**: 핵심 통합 브리지 (3-5일) - 가장 중요
2. **Phase 2**: Langfuse v2 통합 (2-3일) - 추적 시스템
3. **Phase 3**: 전문가 시나리오 (2-3일) - 핵심 기능
4. **Phase 5**: E2E 테스트 (2-3일) - 품질 보증

### ⚡ Parallel Development (병렬 개발 가능)
- Phase 3 (전문가 시나리오) ↔ Phase 4 (시스템 관리)
- Phase 2 (Langfuse) ↔ Phase 4 (성능 최적화)

### 🎯 Milestone 검증 포인트
- **Milestone 1**: Universal Engine Bridge 완성 (Phase 1)
- **Milestone 2**: ChatGPT 스타일 UI 동작 (Phase 1)
- **Milestone 3**: Langfuse v2 추적 시스템 완성 (Phase 2)
- **Milestone 4**: 전문가 시나리오 완벽 동작 (Phase 3)
- **Milestone 5**: E2E 테스트 모든 시나리오 통과 (Phase 5)
- **Milestone 6**: 프로덕션 배포 준비 완료 (Phase 6)

## 🔧 개발 환경 및 도구

### 필수 기술 스택
- **Python**: 3.9+ (기존 환경 유지)
- **Streamlit**: ChatGPT 스타일 UI 구현
- **LLM**: qwen3-4b-fast (Ollama) - 검증된 성능
- **A2A SDK**: 0.2.9 (기존 통합 유지)
- **Langfuse**: v2 (세션 추적)
- **Playwright**: MCP 통합 E2E 테스트

### 검증 도구
- **Backend Verification**: 기존 검증 시스템 활용
- **UI Testing**: Streamlit 테스트 프레임워크
- **E2E Testing**: Playwright MCP 자동화
- **Performance Testing**: 45초 응답 시간 벤치마크

## 🎉 예상 완성 결과

### 기능적 완성도
- **백엔드 보존**: 검증된 Universal Engine 100% 무결성 유지 ✅
- **UI/UX 통합**: ChatGPT 수준의 사용자 경험 제공 ✅
- **기능 확장**: Langfuse v2, SSE 스트리밍, E2E 테스트 100% 구현 ✅

### 성능 및 품질
- **응답 시간**: 평균 45초 유지 (검증된 성능) ✅
- **품질 점수**: 0.8/1.0 달성 ✅
- **시스템 안정성**: 99.9% 가용성 ✅
- **사용자 만족도**: ChatGPT 수준의 UX 제공 ✅

### 전문가 시나리오 지원
- **도메인 분석**: 반도체 이온주입 공정 완벽 분석 ✅
- **전문 해석**: 20년 경력 엔지니어 수준의 인사이트 제공 ✅
- **실무 적용**: 구체적인 조치 방안 및 기술적 권장사항 제공 ✅

## 🌟 최종 목표

**검증된 백엔드 + ChatGPT 스타일 UI = 완벽한 CherryAI 시스템!**

현재 검증된 LLM-First Universal Engine을 100% 보존하면서:
- ✅ ChatGPT 수준의 직관적 사용자 인터페이스
- ✅ 실시간 스트리밍 및 멀티 에이전트 협업 시각화
- ✅ 전문가 시나리오 완벽 지원 (이온주입 + query.txt)
- ✅ 종합적인 추적 및 모니터링 시스템
- ✅ 완전한 E2E 테스트 및 품질 보증

**총 예상 소요 시간: 12-18일**
**검증된 백엔드의 안정성과 성능을 유지하면서 최고 수준의 사용자 경험을 제공하는 완전한 시스템 달성!**