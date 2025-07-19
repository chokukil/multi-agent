# Requirements Document

## Introduction

CherryAI는 A2A 프로토콜 기반의 멀티 에이전트 데이터 분석 플랫폼으로, 사내 모든 구성원이 데이터 분석 전문 지식 없이도 고품질의 데이터 분석을 수행할 수 있도록 지원하는 시스템입니다. 현재 ai.py 기반의 Streamlit 애플리케이션을 cherry_ai.py로 개선하여, ChatGPT Data Analyst와 유사한 사용자 경험을 제공하면서도 멀티 에이전트 협업 과정의 투명성과 신뢰성을 강화하는 것이 목표입니다.

## Requirements

### Requirement 1

**User Story:** 데이터 분석 비전문가로서, 복잡한 데이터 분석 요청을 자연어로 입력하여 전문가 수준의 분석 결과를 얻고 싶다.

#### Acceptance Criteria

1. WHEN 사용자가 자연어로 데이터 분석 요청을 입력 THEN 시스템은 LLM을 통해 요청을 해석하고 적절한 분석 계획을 수립 SHALL
2. WHEN 사용자가 CSV, Excel 등의 데이터 파일을 업로드 THEN 시스템은 자동으로 데이터 구조를 파악하고 분석 가능한 형태로 처리 SHALL
3. WHEN 분석이 완료 THEN 시스템은 수치 기반의 정확한 데이터와 시각화 결과를 제공 SHALL
4. IF 사용자가 추가 질문을 입력 THEN 시스템은 기존 분석 결과를 바탕으로 연속적인 대화형 분석을 수행 SHALL

### Requirement 2

**User Story:** 데이터 분석 전문가로서, 멀티 에이전트가 어떤 과정으로 분석을 수행하는지 투명하게 확인하고 신뢰할 수 있는 결과를 얻고 싶다.

#### Acceptance Criteria

1. WHEN 오케스트레이터가 분석 계획을 수립 THEN 시스템은 어떤 에이전트가 선택되었는지와 그 이유를 명확히 표시 SHALL
2. WHEN 각 에이전트가 작업을 수행 THEN 시스템은 실시간으로 진행 상황을 한 줄 요약으로 표시 SHALL
3. WHEN 사용자가 "View All" 버튼을 클릭 THEN 시스템은 에이전트 선택 이유, 생성된 코드, 중간 결과 등 상세 정보를 표시 SHALL
4. WHEN 분석이 완료 THEN 시스템은 실제 실행된 코드와 그 결과를 함께 제공 SHALL

### Requirement 3

**User Story:** 시스템 관리자로서, A2A 프로토콜 기반의 멀티 에이전트 시스템이 안정적으로 작동하는지 모니터링하고 관리하고 싶다.

#### Acceptance Criteria

1. WHEN 애플리케이션이 시작 THEN 시스템은 모든 A2A 에이전트의 상태를 확인하고 표시 SHALL
2. WHEN 에이전트가 오프라인 상태 THEN 시스템은 해당 에이전트를 제외하고 대안 계획을 수립 SHALL
3. WHEN 에이전트 간 통신 오류가 발생 THEN 시스템은 오류를 기록하고 사용자에게 적절한 피드백을 제공 SHALL
4. IF 시스템 성능이 저하 THEN 시스템은 자동으로 최적화를 수행하고 관리자에게 알림을 제공 SHALL

### Requirement 4

**User Story:** 일반 사용자로서, ChatGPT Data Analyst와 유사한 직관적인 UI/UX를 통해 쉽게 데이터 분석을 수행하고 싶다.

#### Acceptance Criteria

1. WHEN 사용자가 애플리케이션에 접속 THEN 시스템은 ChatGPT와 유사한 채팅 인터페이스를 제공 SHALL
2. WHEN 사용자가 데이터를 업로드 THEN 시스템은 드래그 앤 드롭 방식으로 간편한 업로드 기능을 제공 SHALL
3. WHEN 분석 결과가 생성 THEN 시스템은 텍스트, 차트, 테이블 등을 적절히 조합하여 가독성 높은 결과를 표시 SHALL
4. WHEN 사용자가 결과를 다운로드하려고 할 때 THEN 시스템은 다양한 형식(CSV, PNG, HTML 등)으로 내보내기 기능을 제공 SHALL

### Requirement 5

**User Story:** 반도체 엔지니어로서, 도메인 특화된 분석 요청(예: 이온주입 공정 분석)을 수행하고 전문적인 해석을 얻고 싶다.

#### Acceptance Criteria

1. WHEN 사용자가 도메인 특화 데이터를 업로드 THEN 시스템은 데이터 특성을 파악하고 적절한 전문 에이전트를 선택 SHALL
2. WHEN 복잡한 도메인 지식이 필요한 질문을 입력 THEN 시스템은 해당 도메인의 전문 지식을 활용하여 분석을 수행 SHALL
3. WHEN 분석 결과가 생성 THEN 시스템은 도메인 전문가가 이해할 수 있는 수준의 기술적 해석과 조치 방향을 제공 SHALL
4. IF 이상 패턴이 감지 THEN 시스템은 원인 분석과 함께 실무진이 참고할 수 있는 구체적인 조치 방안을 제안 SHALL

### Requirement 6

**User Story:** 개발자로서, 기존 ai.py의 방대한 코드를 모듈화하여 유지보수성을 높이고 확장 가능한 구조로 개선하고 싶다.

#### Acceptance Criteria

1. WHEN 새로운 cherry_ai.py를 개발 THEN 시스템은 기능별로 모듈화된 구조를 가져야 SHALL
2. WHEN 새로운 에이전트를 추가 THEN 시스템은 최소한의 코드 변경으로 통합이 가능해야 SHALL
3. WHEN 코드를 수정 THEN 시스템은 다른 모듈에 영향을 주지 않는 독립적인 구조를 유지해야 SHALL
4. IF 성능 최적화가 필요 THEN 시스템은 각 모듈별로 독립적인 최적화가 가능해야 SHALL

### Requirement 7

**User Story:** 사용자로서, 데이터를 업로드하면 어떤 분석을 수행하면 좋을지 지능적인 추천을 받고, 분석 완료 후에도 자연스러운 후속 분석 제안을 받고 싶다.

#### Acceptance Criteria

1. WHEN 사용자가 데이터를 업로드 THEN 시스템은 데이터 특성을 분석하여 최대 3개의 추천 분석을 간략한 한 문장으로 버튼 형태로 제안 SHALL
2. WHEN 사용자가 분석 요청에 대한 답변을 완료 THEN 시스템은 LLM이 이전 분석 결과를 바탕으로 다음 단계 분석을 최대 3개 제안 SHALL
3. WHEN 추천 분석이 제안 THEN 각 추천은 사용자가 클릭 한 번으로 바로 실행할 수 있는 형태여야 SHALL
4. IF 데이터 특성이 복잡하거나 다양한 분석이 가능 THEN 시스템은 우선순위를 고려하여 가장 유용한 분석부터 추천해야 SHALL

### Requirement 8

**User Story:** 품질 관리자로서, LLM First 원칙에 따라 하드코딩이나 패턴 매칭 없이 범용적이면서도 정확한 분석 결과를 보장하고 싶다.

#### Acceptance Criteria

1. WHEN 시스템이 분석을 수행 THEN 모든 의사결정은 LLM을 통해 이루어져야 하며 하드코딩된 규칙이 없어야 SHALL
2. WHEN 새로운 유형의 데이터나 질문이 입력 THEN 시스템은 패턴 매칭 없이 LLM의 추론 능력으로 처리해야 SHALL
3. WHEN 분석 결과를 검증 THEN 시스템은 LLM 기반의 자체 검증 메커니즘을 통해 결과의 신뢰성을 확보해야 SHALL
4. IF 예상치 못한 상황이 발생 THEN 시스템은 LLM의 일반화 능력을 활용하여 적절히 대응해야 SHALL