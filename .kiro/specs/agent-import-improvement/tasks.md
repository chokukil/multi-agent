# Agent Import Improvement Implementation Plan

## 진행 현황 개요

- ✅ **완료**: `data_cleaning_server_clean.py` 성공 패턴 확립
- 🔄 **진행 중**: 전체 시스템 개선 계획 수립
- ⏳ **대기 중**: 9개 서버 마이그레이션 및 공통 모듈 구축

---

## Phase 1: 기반 구조 구축

### ✅ 1.1 패키지 구조 정규화 (완료)
- ai_data_science_team 패키지를 프로젝트 루트로 이동 완료
- 상대 import 문제의 근본적 해결 기반 마련
- 성공 패턴 검증 완료 (data_cleaning_server_clean.py)
- _Requirements: 1.1, 2.1_

### ⏳ 1.2 공통 모듈 디렉토리 생성
- `a2a_ds_servers/common/` 디렉토리 구조 생성
- `__init__.py` 파일들 생성하여 패키지 구조 완성
- 공통 모듈 기반 인프라 구축
- _Requirements: 3.1, 3.2_

### ⏳ 1.3 Import 유틸리티 모듈 구현
- `a2a_ds_servers/common/import_utils.py` 생성
- `setup_project_paths()` 함수 구현
- `safe_import_ai_ds_team()` 함수 구현
- `get_ai_ds_agent()` 함수 구현
- 모든 서버에서 사용할 표준 import 패턴 정의
- _Requirements: 2.1, 2.2, 5.3_

### ⏳ 1.4 공통 서버 베이스 클래스 구현
- `a2a_ds_servers/common/base_server.py` 생성
- `BaseA2AServer` 추상 클래스 구현
- 표준 AgentCard 생성 로직 구현
- 공통 서버 실행 로직 구현
- _Requirements: 3.1, 3.2, 8.1_

### ⏳ 1.5 공통 데이터 처리기 구현
- `a2a_ds_servers/common/data_processor.py` 생성
- `CommonDataProcessor` 클래스 구현
- CSV/JSON 파싱 로직 표준화
- 샘플 데이터 생성 로직 통합
- _Requirements: 3.2, 7.1_

---

## Phase 2: 상대 Import 문제 완전 해결

### ⏳ 2.1 남은 상대 import 문제 식별
- `ai_data_science_team` 패키지 내 모든 상대 import 스캔
- 문제가 되는 2단계 상대 import (`from ..`) 목록 작성
- 각 모듈별 의존성 관계 매핑
- _Requirements: 2.1, 2.2_

### ⏳ 2.2 agents 모듈 상대 import 수정
- `agents/data_wrangling_agent.py` 상대 import 수정
- 기타 agents 모듈의 상대 import 검토 및 수정
- 수정 후 import 테스트 실행
- _Requirements: 2.1, 2.2_

### ⏳ 2.3 tools 모듈 상대 import 수정
- ✅ `tools/eda.py` 수정 완료
- 기타 tools 모듈의 상대 import 검토
- 모든 tools 모듈 import 테스트 실행
- _Requirements: 2.1, 2.2_

### ⏳ 2.4 전체 패키지 import 검증
- 모든 모듈의 import 성공 여부 자동 테스트 스크립트 작성
- CI/CD 파이프라인에 import 테스트 추가
- 문제 발생 시 자동 알림 시스템 구축
- _Requirements: 2.1, 2.2, 5.1_

---

## Phase 3: 기존 서버 Import 구조 개선

### ⏳ 3.1 기존 서버 Import 상태 분석
- 현재 a2a_ds_servers 디렉토리의 모든 서버 파일 스캔
- 각 서버별 import 오류 및 의존성 문제 파악
- 루트 이동 후 자동으로 해결된 부분과 수동 수정 필요 부분 구분
- _Requirements: 2.1, 5.1_

### ⏳ 3.2 서버별 sys.path 설정 표준화
- 모든 서버 파일의 sys.path 설정을 표준 패턴으로 통일
- 불필요한 복잡한 경로 설정 제거
- 공통 import_utils 모듈 사용으로 변경
- _Requirements: 1.1, 1.2_

### ⏳ 3.3 서버별 AI DS Team 모듈 활용 최적화
- 각 서버가 필요한 AI DS Team 모듈만 선택적으로 import
- 폴백 메커니즘 구현으로 안정성 향상
- 원본 기능 최대한 활용하도록 개선
- _Requirements: 4.1, 4.2, 4.3_

---

## Phase 4: 통합 테스트 및 검증

### ⏳ 4.1 개별 서버 단위 테스트
- 각 서버별 독립 실행 테스트 스크립트 작성
- Import 성공 여부 자동 검증
- 기본 기능 동작 테스트
- 성능 벤치마크 측정
- _Requirements: 5.1, 5.2, 7.1_

### ⏳ 4.2 서버 간 통합 테스트
- 여러 서버 동시 실행 테스트
- 메모리 사용량 모니터링
- 포트 충돌 방지 검증
- 리소스 공유 테스트
- _Requirements: 5.1, 7.2_

### ⏳ 4.3 End-to-End 워크플로우 테스트
- 실제 데이터 사이언스 워크플로우 시뮬레이션
- 데이터 로딩 → 클리닝 → 분석 → 시각화 파이프라인 테스트
- 오류 복구 시나리오 테스트
- 사용자 시나리오 기반 테스트
- _Requirements: 5.1, 5.2, 5.3_

### ⏳ 4.4 성능 최적화 및 튜닝
- 서버 시작 시간 최적화
- 메모리 사용량 최적화
- Import 시간 단축
- 불필요한 모듈 로딩 제거
- _Requirements: 7.1, 7.2_

---

## Phase 5: 문서화 및 가이드라인

### ⏳ 5.1 개발자 가이드 작성
- 새로운 서버 추가 가이드라인
- Import 패턴 사용법 문서화
- 공통 모듈 활용 방법 설명
- 트러블슈팅 가이드 작성
- _Requirements: 8.1, 8.2_

### ⏳ 5.2 API 문서 업데이트
- 각 서버별 API 스펙 문서화
- AgentCard 정보 표준화
- 예제 요청/응답 업데이트
- Postman 컬렉션 생성
- _Requirements: 8.1, 8.2_

### ⏳ 5.3 운영 가이드 작성
- 서버 배포 가이드
- 모니터링 및 로깅 설정
- 오류 대응 매뉴얼
- 성능 튜닝 가이드
- _Requirements: 6.1, 6.2, 8.3_

### ⏳ 5.4 테스트 자동화 구축
- CI/CD 파이프라인에 테스트 통합
- 자동 회귀 테스트 설정
- 성능 모니터링 대시보드 구축
- 알림 시스템 설정
- _Requirements: 5.1, 6.3_

---

## Phase 6: 최종 검증 및 배포

### ⏳ 6.1 프로덕션 환경 테스트
- 실제 프로덕션 환경에서 테스트
- 부하 테스트 실행
- 장애 복구 테스트
- 보안 검증
- _Requirements: 6.1, 6.2_

### ⏳ 6.2 사용자 수용 테스트
- 실제 사용자 시나리오 테스트
- 피드백 수집 및 반영
- 사용성 개선
- 최종 버그 수정
- _Requirements: 5.2, 8.1_

### ⏳ 6.3 배포 및 모니터링
- 단계적 배포 (Blue-Green 배포)
- 실시간 모니터링 설정
- 롤백 계획 준비
- 사용자 교육 및 지원
- _Requirements: 6.1, 6.2, 6.3_

### ⏳ 6.4 프로젝트 완료 및 정리
- 최종 테스트 리포트 작성
- 성과 측정 및 분석
- 교훈 정리 및 문서화
- 향후 개선 계획 수립
- _Requirements: 8.1, 8.2, 8.3_

---

## 우선순위 및 일정

### 🔥 High Priority (즉시 시작)
1. **Phase 1.2-1.5**: 공통 모듈 구축 (1-2일)
2. **Phase 2.1-2.4**: 상대 import 문제 완전 해결 (1일)
3. **Phase 3.1**: Data Loader Server 마이그레이션 (1일)

### 🟡 Medium Priority (1주 내)
1. **Phase 3.2-3.5**: 핵심 서버들 마이그레이션 (3-4일)
2. **Phase 4.1**: 개별 서버 테스트 (1일)

### 🟢 Low Priority (2주 내)
1. **Phase 3.6-3.9**: 나머지 서버 마이그레이션 (2-3일)
2. **Phase 4.2-4.4**: 통합 테스트 및 최적화 (2-3일)
3. **Phase 5.1-5.4**: 문서화 (2-3일)

### 📋 리스크 및 대응 방안

**리스크 1**: AI DS Team 모듈의 복잡한 의존성
- **대응**: 단계적 마이그레이션 및 폴백 메커니즘 구축

**리스크 2**: 기존 서버 기능 손실
- **대응**: 철저한 테스트 및 기존 버전 백업 유지

**리스크 3**: 성능 저하
- **대응**: 성능 모니터링 및 최적화 단계 포함

---

## 성공 지표

### 기술적 지표
- ✅ 모든 서버 import 오류 0건
- ✅ 서버 시작 시간 < 5초
- ✅ 메모리 사용량 < 200MB (기본)
- ✅ 테스트 커버리지 > 80%

### 운영적 지표
- ✅ 서버 가동률 > 99.9%
- ✅ 평균 응답 시간 < 2초
- ✅ 오류 발생률 < 0.1%

### 개발 생산성 지표
- ✅ 새 서버 추가 시간 < 1시간
- ✅ 버그 수정 시간 50% 단축
- ✅ 코드 중복률 < 10%