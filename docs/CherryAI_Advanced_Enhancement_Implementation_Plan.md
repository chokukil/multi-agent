# 🚀 CherryAI Advanced Enhancement Implementation Plan

## 📋 **현재 진행 상황 및 개선 계획**

### 🎯 **Phase 1: Shared Knowledge Bank 고급 임베딩 검색 시스템**

#### **1.1 기술 스택 선정**
- **벡터 DB**: ChromaDB (로컬 개발 최적화, Python-first)
- **임베딩 모델**: Sentence-BERT (all-MiniLM-L6-v2) - 빠른 로컬 처리
- **검색 전략**: 하이브리드 (Dense + Sparse) 검색
- **청킹 전략**: Context-aware 청킹 (500-1000 토큰)

#### **1.2 핵심 구현 요소**
```python
- AdvancedSharedKnowledgeBank: 메인 클래스
- KnowledgeType: FILE_DATA, AGENT_MEMORY, USER_QUERY, ANALYSIS_RESULT
- SearchStrategy: SEMANTIC, KEYWORD, HYBRID, CONTEXTUAL
- 실시간 지식 동기화 및 A2A 에이전트별 분할
```

#### **1.3 A2A 통합**
- Context Engineering 6 Data Layers와 완전 통합
- 에이전트 간 지식 공유 자동화
- 메타데이터 enrichment

---

## 🧠 **Phase 2: LLM First 원칙 완전 준수**

### **2.1 핵심 원칙**
- ❌ **절대 금지**: Rule 기반 하드코딩, 패턴 매칭, 템플릿 매칭
- ✅ **필수 준수**: LLM 능력 최대 활용, 범용적 동작, 사용자 의도 반영
- ✅ **A2A 표준**: SDK 0.2.9, SSE 스트리밍, ASYNC 방식

### **2.2 하드코딩 제거 대상**
1. **특정 데이터셋 전용 로직** (타이타닉, 아이리스 등)
2. **고정된 컬럼명 참조** ('Survived', 'Sex', 'Pclass' 등)
3. **Rule 기반 데이터 타입 판단**
4. **패턴 매칭 기반 쿼리 처리**

### **2.3 LLM First 구현 방식**
```python
# ❌ 하드코딩 방식
if 'Survived' in columns:
    return classification_analysis()

# ✅ LLM First 방식  
context = f"데이터 컬럼: {columns}, 사용자 요청: {user_query}"
analysis_type = await llm.determine_analysis_type(context)
return await llm.generate_analysis(data, analysis_type, user_query)
```

---

## 🧪 **Phase 3: 체계적 테스트 전략**

### **3.1 테스트 계층**

#### **Level 1: 단위 테스트 (pytest)**
```bash
# 개별 컴포넌트 테스트
pytest tests/unit/test_shared_knowledge_bank.py
pytest tests/unit/test_llm_first_compliance.py
pytest tests/unit/test_a2a_agents.py
pytest tests/unit/test_mcp_tools.py
```

#### **Level 2: 통합 테스트 (pytest)**
```bash
# 컴포넌트 간 상호작용 테스트
pytest tests/integration/test_a2a_mcp_integration.py
pytest tests/integration/test_knowledge_bank_integration.py
pytest tests/integration/test_streaming_integration.py
```

#### **Level 3: E2E 테스트 (Playwright MCP)**
```bash
# 실제 UI 및 사용자 시나리오 테스트
- 데이터 업로드 → 분석 → 결과 확인
- A2A 에이전트 체인 실행
- MCP 도구 연동 검증
- 실시간 스트리밍 검증
```

### **3.2 검증 기준**

#### **기능적 검증**
- ✅ 모든 A2A 에이전트 (11개) 정상 동작
- ✅ 모든 MCP 도구 (7개) 연동 성공
- ✅ 실시간 SSE 스트리밍 정상
- ✅ 지식 뱅크 임베딩 검색 정확도

#### **품질적 검증**
- ✅ 결과의 정확성 (입력 대비 적절한 분석)
- ✅ LLM First 원칙 준수 (하드코딩 0%)
- ✅ 응답 시간 (<500ms)
- ✅ 오류 복구 능력

---

## 🏗️ **Phase 4: main.py 모듈화 전략**

### **4.1 현재 main.py 분석**
- 크기: ~500 라인 (임계점 근접)
- 문제: UI 로직과 비즈니스 로직 혼재
- 해결: 컴포넌트 기반 모듈 분리

### **4.2 모듈 분리 계획**
```python
main.py                     # 진입점만 (50 라인 이하)
├── ui/main_ui_controller.py     # UI 컨트롤러
├── core/main_app_engine.py      # 비즈니스 엔진  
├── core/main_config_manager.py  # 설정 관리
└── core/main_session_manager.py # 세션 관리
```

### **4.3 의존성 주입 패턴**
- 각 모듈은 독립적으로 테스트 가능
- Interface 기반 loose coupling
- Factory 패턴으로 인스턴스 생성

---

## 🎯 **Phase 5: 실행 순서**

### **Step 1: 문서화 및 계획 수립** ✅
- 현재 단계 완료

### **Step 2: Shared Knowledge Bank 완성**
```bash
1. ChromaDB 의존성 설치
2. 고급 임베딩 검색 구현
3. A2A 통합 테스트
4. 성능 최적화
```

### **Step 3: LLM First 개선**
```bash
1. 하드코딩 감지 및 제거
2. LLM 기반 동적 분석 로직
3. 범용적 데이터 처리 엔진
4. 검증 테스트
```

### **Step 4: main.py 모듈화**
```bash
1. 컴포넌트 분리
2. Interface 정의
3. 의존성 주입 구현
4. 통합 테스트
```

### **Step 5: 종합 테스트**
```bash
1. pytest 단위/통합 테스트
2. Playwright MCP E2E 테스트
3. 성능 및 품질 검증
4. 최종 문서화
```

---

## 🔍 **테스트 매트릭스**

| 컴포넌트 | 단위테스트 | 통합테스트 | E2E테스트 | 품질검증 |
|----------|-----------|----------|----------|----------|
| **A2A 에이전트 (11개)** | pytest | pytest | Playwright | LLM 평가 |
| **MCP 도구 (7개)** | pytest | pytest | Playwright | 결과 검증 |
| **Shared Knowledge Bank** | pytest | pytest | Playwright | 검색 정확도 |
| **실시간 스트리밍** | pytest | pytest | Playwright | 지연시간 |
| **UI 컴포넌트** | - | pytest | Playwright | UX 검증 |
| **LLM First 준수** | pytest | pytest | Playwright | 하드코딩 0% |

---

## 📊 **성공 기준**

### **필수 기준**
- [ ] 모든 테스트 통과율 95% 이상
- [ ] 하드코딩 패턴 0개 검출
- [ ] 응답 시간 500ms 이하
- [ ] 메모리 사용량 최적화

### **품질 기준**  
- [ ] 분석 결과 정확도 90% 이상
- [ ] 사용자 의도 반영도 95% 이상
- [ ] A2A 에이전트 체인 성공률 98% 이상
- [ ] 지식 뱅크 검색 정확도 85% 이상

---

## 🚨 **주의사항**

1. **Playwright MCP 연결 실패시**:
   - MCP 서버 비활성화 → 대기 → 재활성화
   - 절대 다른 대안 도구 사용 금지

2. **오류 발생시**:
   - 증상 해결보다 근본 원인 파악 우선
   - LLM First 원칙 위반 여부 확인
   - 임시방편 솔루션 절대 금지

3. **테스트 실패시**:
   - 단순 통과보다 결과 품질 중요
   - 사용자 관점에서 정확성 검증
   - A2A + MCP 통합 체인 전체 검증

---

**📝 마지막 업데이트**: 2024-01-20
**📋 담당자**: CherryAI Development Team
**🎯 목표**: 세계 최초 A2A + MCP + Enterprise LLM First 플랫폼 완성 