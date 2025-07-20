# 🧠 LLM First 범용 도메인 분석 엔진 요구사항 명세서

## 📋 개요

이 문서는 진정한 LLM First 범용 도메인 분석 엔진 구현을 위한 상세 요구사항을 정의합니다. 이 시스템은 하드코딩된 패턴, 카테고리, 규칙 없이 모든 도메인(반도체, 금융, 의료 등)과 모든 사용자 수준(초보자~전문가)에 동적으로 적응하는 완전한 범용 분석 엔진입니다.

### 핵심 철학
- **Zero Hardcoding**: 사전 정의된 패턴, 카테고리, 규칙 일체 없음
- **Zero Assumptions**: 도메인, 사용자 수준, 쿼리 유형에 대한 가정 없음  
- **Self-Discovering**: LLM이 스스로 모든 컨텍스트와 요구사항을 파악
- **Universal Adaptability**: 모든 도메인, 모든 사용자 수준에 동적 적응

### 설계 목표
1. **완전한 범용성**: 반도체부터 금융, 의료까지 모든 도메인 지원
2. **사용자 적응성**: 초보자부터 전문가까지 자동 수준 조절
3. **진정한 지능**: 패턴 매칭이 아닌 실제 이해와 추론
4. **지속적 학습**: 상호작용을 통한 실시간 개선

## 🔍 기존 문제점 분석 및 해결 요구사항

### ❌ 제거해야 할 잘못된 접근법들

#### 하드코딩된 분류 체계 제거
```python
# 제거해야 할 잘못된 패턴들
if "도즈" in query or "균일성" in query:
    process_type = "ion_implantation"
    analysis_category = "dose_uniformity"

domain_categories = {
    "semiconductor": ["ion_implantation", "lithography"],
    "finance": ["risk_analysis", "portfolio"],
    "healthcare": ["diagnosis", "treatment"]
}

if user_type == "expert":
    use_technical_language()
elif user_type == "beginner":
    use_simple_language()
```

## Requirements

### Requirement 1: Zero Hardcoding Architecture Implementation

**User Story:** As a system architect, I want to completely eliminate all hardcoded patterns, categories, and domain-specific logic, so that the system can truly adapt to any domain without predefined limitations.

#### Acceptance Criteria

1. WHEN processing any query THEN the system SHALL NOT use predefined domain categories like `if "도즈" in query or "균일성" in query: process_type = "ion_implantation"`
2. WHEN analyzing data THEN the system SHALL NOT rely on hardcoded classification systems like `domain_categories = {"semiconductor": ["ion_implantation", "lithography"]}`
3. WHEN determining user expertise THEN the system SHALL NOT use fixed persona categories like `if user_type == "expert": use_technical_language()`
4. WHEN implementing UniversalQueryProcessor THEN the system SHALL use the architecture:
   ```python
   class UniversalQueryProcessor:
       """완전 범용 쿼리 처리기 - 어떤 가정도 하지 않음"""
       
       async def process_query(self, query: str, data: Any, context: Dict = None):
           """
           순수 LLM 기반으로 쿼리 처리
           - 패턴 매칭 없음
           - 사전 분류 없음  
           - 완전한 동적 분석
           """
   ```
5. WHEN encountering new domains THEN the system SHALL adapt without requiring code changes or configuration updates

### Requirement 2: Meta-Reasoning Engine with DeepSeek-R1 Inspired Patterns

**User Story:** As a user, I want the system to think about its own thinking process using 2024-2025 최신 연구 기반 meta-reasoning, so that I receive increasingly sophisticated and accurate analysis.

#### Acceptance Criteria

1. WHEN analyzing any query THEN the system SHALL implement MetaReasoningEngine with methods:
   ```python
   class MetaReasoningEngine:
       """메타 추론 엔진 - 생각에 대해 생각하기"""
       
       async def analyze_query_intent(self, query: str, data: Any):
           """쿼리 의도를 스스로 파악"""
           
       async def detect_domain_context(self, query: str, data: Any):
           """도메인 컨텍스트를 스스로 발견"""
           
       async def estimate_user_expertise(self, interaction_history: List):
           """사용자 전문성을 상호작용으로 추정"""
           
       async def select_response_strategy(self, intent, context, expertise):
           """최적 응답 전략을 스스로 선택"""
   ```

2. WHEN performing self-reflection THEN the system SHALL use the exact prompt pattern:
   ```
   # 자가 반성 추론 패턴
   당신은 주어진 쿼리와 데이터를 분석하는 전문가입니다.

   단계 1: 초기 관찰
   - 데이터를 보고 무엇을 발견하는가?
   - 사용자 쿼리의 진정한 의도는?
   - 내가 놓치고 있는 것은 없는가?

   단계 2: 다각도 분석
   - 이 문제를 다른 방식으로 접근한다면?
   - 사용자가 전문가라면 어떤 답을 원할까?
   - 사용자가 초보자라면 어떤 도움이 필요할까?

   단계 3: 자가 검증
   - 내 분석이 논리적으로 일관성이 있는가?
   - 사용자에게 실제로 도움이 되는가?
   - 확신이 없는 부분은 무엇인가?

   단계 4: 적응적 응답
   - 확실한 부분은 명확히 제시
   - 불확실한 부분은 명확화 질문
   - 사용자 수준에 맞는 설명 깊이 조절
   ```

3. WHEN evaluating analysis quality THEN the system SHALL use meta-rewarding pattern:
   ```
   # 자가 평가 및 개선 패턴
   내 분석을 스스로 평가해보겠습니다:

   평가 기준:
   1. 정확성: 분석이 데이터를 올바르게 해석했는가?
   2. 완전성: 중요한 인사이트를 놓치지 않았는가?
   3. 적절성: 사용자 수준과 요구에 맞는가?
   4. 명확성: 설명이 이해하기 쉬운가?
   5. 실용성: 실제로 도움이 되는 조치를 제안했는가?

   개선점:
   - 부족한 부분은 무엇인가?
   - 어떻게 더 나은 분석을 할 수 있는가?
   - 사용자에게 추가로 필요한 정보는?

   이 평가를 바탕으로 응답을 개선하겠습니다.
   ```

4. WHEN uncertainty exists THEN the system SHALL explicitly state "확신이 없는 부분은 무엇인가?" and seek clarification
5. WHEN receiving user feedback THEN the system SHALL incorporate feedback into its meta-reasoning process for future improvements

### Requirement 3: Dynamic Context Discovery

**User Story:** As a user working with any type of data, I want the system to automatically discover the domain context and requirements from the data itself, so that I don't need to explain what type of analysis I need.

#### Acceptance Criteria

1. WHEN receiving data and query THEN the system SHALL analyze data characteristics, patterns, and terminology to discover domain context
2. WHEN domain context is unclear THEN the system SHALL use the pattern: "이 데이터를 보니 뭔가 공장에서 제품을 만드는 과정을 기록한 것 같네요" for intuitive explanation
3. WHEN discovering domain patterns THEN the system SHALL identify relevant methodologies and best practices without predefined knowledge bases
4. WHEN context discovery is incomplete THEN the system SHALL ask targeted clarifying questions to complete understanding
5. WHEN new domains are encountered THEN the system SHALL learn domain-specific patterns through real-time interaction

### Requirement 4: Adaptive User Understanding

**User Story:** As a user of any expertise level, I want the system to understand my knowledge level through our interaction and adapt its communication style accordingly, so that I receive appropriately tailored explanations.

#### Acceptance Criteria

1. WHEN user submits first query THEN the system SHALL analyze language usage, terminology, question complexity to estimate expertise level
2. WHEN expertise estimation is uncertain THEN the system SHALL use progressive disclosure to gauge user understanding level
3. WHEN user is identified as beginner THEN the system SHALL use patterns like "마치 요리 레시피의 재료 분량을 측정한 기록처럼 보여요" for accessible explanations
4. WHEN user is identified as expert THEN the system SHALL provide technical analysis like "Cpk 1.2에서 1.33으로 개선하려면 변동성을 약 8.3% 감소시켜야 합니다"
5. WHEN user expertise changes during conversation THEN the system SHALL dynamically adjust explanation depth and technical language usage

### Requirement 5: Self-Reflecting Reasoning System

**User Story:** As a user, I want the system to continuously question and improve its own analysis process, so that I can trust the quality and reliability of the insights provided.

#### Acceptance Criteria

1. WHEN performing analysis THEN the system SHALL use DeepSeek-R1 inspired reasoning with steps: 초기 관찰 → 다각도 분석 → 자가 검증 → 적응적 응답
2. WHEN generating insights THEN the system SHALL apply meta-rewarding patterns to evaluate accuracy, completeness, appropriateness, clarity, and practicality
3. WHEN confidence is low THEN the system SHALL explicitly state "확신이 없는 부분은 무엇인가?" and seek clarification
4. WHEN multiple reasoning paths exist THEN the system SHALL use chain-of-thought with self-consistency to validate conclusions
5. WHEN analysis is complete THEN the system SHALL perform final self-reflection: "이 평가를 바탕으로 응답을 개선하겠습니다"

### Requirement 6: Universal Intent Detection

**User Story:** As a user with various types of questions, I want the system to understand my true intent without forcing me into predefined categories, so that I can ask questions naturally and get relevant responses.

#### Acceptance Criteria

1. WHEN processing queries THEN the system SHALL use semantic routing without predefined intent categories
2. WHEN intent is unclear THEN the system SHALL apply the pattern: "사전 정의된 카테고리나 패턴에 의존하지 않고 쿼리 자체가 말하는 것을 들어보겠습니다"
3. WHEN analyzing user intent THEN the system SHALL distinguish between direct intent (명시적 요청) and implicit intent (암묵적 의도)
4. WHEN multiple interpretations exist THEN the system SHALL explore semantic space navigation to find the most relevant approach
5. WHEN intent detection is complete THEN the system SHALL select response strategy based on discovered intent rather than predefined templates

### Requirement 7: Progressive Disclosure System

**User Story:** As a user, I want the system to reveal information gradually based on my interest and understanding level, so that I'm neither overwhelmed with too much detail nor left wanting more depth.

#### Acceptance Criteria

1. WHEN providing initial response THEN the system SHALL present core insights first with clear options to explore deeper
2. WHEN user shows confusion THEN the system SHALL simplify using patterns like "일단 몇 가지 흥미로운 패턴이 보이는데요" with accessible explanations
3. WHEN user demonstrates understanding THEN the system SHALL offer more technical depth and detailed analysis
4. WHEN user asks follow-up questions THEN the system SHALL adapt disclosure level based on question sophistication
5. WHEN analysis is complete THEN the system SHALL suggest relevant next steps: "어떤 부분이 가장 궁금하세요?"

### Requirement 8: Zero-Shot Adaptive Reasoning

**User Story:** As a user with unique or novel problems, I want the system to reason about my situation without relying on previous templates or examples, so that I get fresh insights tailored to my specific context.

#### Acceptance Criteria

1. WHEN encountering new problem types THEN the system SHALL use zero-shot reasoning without template matching
2. WHEN reasoning about problems THEN the system SHALL follow the pattern: 문제 공간 정의 → 추론 전략 수립 → 단계별 추론 실행 → 결과 통합 및 검증
3. WHEN assumptions are made THEN the system SHALL explicitly state "가정과 제약사항 명시" and "불확실성과 신뢰도 평가"
4. WHEN multiple reasoning paths exist THEN the system SHALL validate consistency across different approaches
5. WHEN reasoning is complete THEN the system SHALL state "템플릿이나 공식에 의존하지 않고 문제 자체의 본질에 맞는 추론을 수행하겠습니다"

### Requirement 9: Real-time Learning and Adaptation

**User Story:** As a user, I want the system to learn from our interactions and improve its responses over time, so that the quality of analysis gets better with continued use.

#### Acceptance Criteria

1. WHEN user provides feedback THEN the system SHALL incorporate learning using the pattern: "이번 상호작용에서 배운 것을 정리하겠습니다"
2. WHEN successful patterns are identified THEN the system SHALL generalize learnings for similar future situations
3. WHEN failures occur THEN the system SHALL analyze failure patterns and avoid similar approaches
4. WHEN user satisfaction changes THEN the system SHALL adjust its approach based on satisfaction indicators
5. WHEN knowledge is updated THEN the system SHALL maintain learning without compromising user privacy

### Requirement 10: Cherry AI UI/UX Integration and Enhancement

**User Story:** As a Cherry AI user, I want a ChatGPT-like intuitive interface that seamlessly integrates the Universal Engine with A2A agents, so that I get an enhanced multi-agent data analysis experience without losing the familiar UI/UX.

#### Acceptance Criteria

1. WHEN using Cherry AI THEN the system SHALL maintain the existing ChatGPT-style interface with these components:
   ```python
   # 기존 Cherry AI UI 컴포넌트 유지 및 강화
   class CherryAI:
       def render_header(self):
           """🍒 Cherry AI 브랜딩 헤더 유지 + Universal Engine 상태 표시"""
           st.markdown("# 🍒 Cherry AI - LLM First Universal Engine")
           
           # Universal Engine 상태 표시
           col1, col2, col3 = st.columns([2, 1, 1])
           with col1:
               st.caption("🧠 Universal Engine 활성화")
           with col2:
               st.caption(f"🤖 {len(self.available_agents)}개 A2A 에이전트")
           with col3:
               if st.button("⚙️ 설정"):
                   st.session_state.show_settings = True
           
       def render_chat_interface(self):
           """💬 ChatGPT 스타일 채팅 인터페이스 유지 + 메타 추론 표시"""
           # 기존 채팅 메시지 표시
           for message in st.session_state.messages:
               with st.chat_message(message["role"]):
                   st.write(message["content"])
                   
                   # Universal Engine 메타 추론 결과 표시
                   if message.get("meta_reasoning"):
                       with st.expander("🧠 메타 추론 과정", expanded=False):
                           st.json(message["meta_reasoning"])
                   
                   # A2A Agent 기여도 표시
                   if message.get("agent_contributions"):
                       with st.expander("🤖 에이전트 협업", expanded=False):
                           for agent_id, contribution in message["agent_contributions"].items():
                               st.write(f"**{agent_id}**: {contribution}")
           
       def render_file_upload(self):
           """📁 직관적 파일 업로드 인터페이스 + Universal Engine 데이터 분석"""
           uploaded_file = st.file_uploader(
               "📁 데이터 파일 업로드", 
               type=['csv', 'xlsx', 'json', 'txt'],
               help="Universal Engine이 자동으로 데이터 유형과 도메인을 감지합니다"
           )
           
           if uploaded_file:
               # Universal Engine으로 데이터 사전 분석
               with st.spinner("🧠 데이터 컨텍스트 분석 중..."):
                   context_analysis = await self.universal_engine.analyze_data_context(uploaded_file)
               
               # 감지된 도메인과 추천 분석 표시
               if context_analysis:
                   st.success(f"✅ {context_analysis['domain']} 도메인 데이터로 감지됨")
                   st.info(f"💡 추천 분석: {context_analysis['recommended_analysis']}")
           
       def render_sidebar(self):
           """🔧 에이전트 상태 및 설정 사이드바 + Universal Engine 제어"""
           with st.sidebar:
               st.header("🔧 Universal Engine 제어")
               
               # 메타 추론 설정
               st.subheader("🧠 메타 추론 설정")
               show_reasoning = st.checkbox("추론 과정 표시", value=True)
               reasoning_depth = st.selectbox("추론 깊이", ["기본", "상세", "전문가"])
               
               # A2A Agent 상태
               st.subheader("🤖 A2A 에이전트 상태")
               for agent in self.available_agents:
                   status_icon = "🟢" if agent.status == "active" else "🔴"
                   st.write(f"{status_icon} {agent.name} (:{agent.port})")
               
               # 사용자 프로필 설정
               st.subheader("👤 사용자 프로필")
               expertise_level = st.selectbox(
                   "전문성 수준", 
                   ["자동 감지", "초보자", "중급자", "전문가"]
               )
               
               # Universal Engine 통계
               st.subheader("📊 엔진 통계")
               st.metric("총 분석 수행", st.session_state.get('total_analyses', 0))
               st.metric("평균 응답 시간", f"{st.session_state.get('avg_response_time', 0):.1f}초")
               st.metric("사용자 만족도", f"{st.session_state.get('satisfaction_score', 0):.1f}/5.0")
   ```

2. WHEN replacing hardcoded analysis logic THEN the system SHALL completely replace the execute_analysis method:
   ```python
   # 제거되는 기존 하드코딩 패턴:
   async def execute_analysis(self, user_query: str) -> Dict[str, Any]:
       # ❌ 제거해야 할 하드코딩된 도메인 우선순위
       if SEMICONDUCTOR_ENGINE_AVAILABLE:
           semiconductor_result = await analyze_semiconductor_data(...)
           if confidence > 0.7:
               return self._format_semiconductor_analysis(semiconductor_result)
       # ❌ 제거해야 할 일반 A2A fallback
       return await self._general_agent_analysis(user_query)
   
   # ✅ 새로운 Universal Engine + A2A 통합 패턴:
   async def execute_analysis(self, user_query: str) -> Dict[str, Any]:
       """완전히 새로운 LLM First 분석 실행"""
       
       # 1. Universal Engine으로 메타 추론 수행
       meta_analysis = await self.universal_engine.perform_meta_reasoning(
           query=user_query,
           data=st.session_state.get('current_data'),
           user_context=self._get_user_context(),
           conversation_history=st.session_state.get('messages', [])
       )
       
       # 2. A2A Agent 동적 선택 및 협업
       unified_result = await self.universal_a2a_system.process_unified_query(
           query=user_query,
           meta_analysis=meta_analysis,
           available_agents=self.available_agents,
           context=self._get_session_context()
       )
       
       # 3. 결과 통합 및 사용자 수준별 적응
       adaptive_response = await self.universal_engine.generate_adaptive_response(
           analysis_result=unified_result,
           user_profile=st.session_state.get('user_profile', {}),
           meta_reasoning=meta_analysis
       )
       
       return adaptive_response
   ```

3. WHEN displaying analysis progress THEN the system SHALL show real-time Universal Engine processing:
   ```python
   # 메타 추론 과정 실시간 표시 (더 상세한 단계별 표시)
   async def display_analysis_progress(self, user_query: str):
       with st.chat_message("assistant"):
           # 1단계: 메타 추론
           with st.spinner("🧠 메타 추론 중..."):
               st.caption("쿼리 의도 분석, 도메인 컨텍스트 발견, 사용자 수준 추정")
               meta_analysis = await self.universal_engine.analyze_request(user_query)
               
           # 메타 추론 결과 미리보기
           if st.session_state.get('show_reasoning', True):
               with st.expander("🧠 메타 추론 결과", expanded=False):
                   st.write(f"**감지된 도메인**: {meta_analysis.get('domain', '분석 중')}")
                   st.write(f"**사용자 수준**: {meta_analysis.get('user_level', '추정 중')}")
                   st.write(f"**분석 전략**: {meta_analysis.get('strategy', '수립 중')}")
           
           # 2단계: A2A 에이전트 선택
           with st.spinner("🤖 A2A 에이전트 선택 중..."):
               st.caption("메타 분석 결과를 바탕으로 최적 에이전트 조합 선택")
               selected_agents = await self.agent_selector.select_agents(meta_analysis)
               
           # 선택된 에이전트 표시
           if selected_agents:
               st.success(f"✅ {len(selected_agents)}개 에이전트 선택됨")
               cols = st.columns(len(selected_agents))
               for i, agent in enumerate(selected_agents):
                   with cols[i]:
                       st.write(f"🤖 {agent.name}")
                       st.caption(f"포트: {agent.port}")
           
           # 3단계: 에이전트 협업 실행
           with st.spinner("⚡ 에이전트 협업 실행 중..."):
               st.caption("선택된 에이전트들이 협업하여 분석 수행")
               
               # 실시간 진행률 표시
               progress_bar = st.progress(0)
               status_text = st.empty()
               
               async for progress_update in self.workflow_coordinator.execute_workflow_with_progress(selected_agents):
                   progress_bar.progress(progress_update['progress'] / 100)
                   status_text.text(f"진행 중: {progress_update['current_task']}")
               
               agent_results = progress_update['final_results']
           
           # 4단계: 결과 통합 및 적응적 응답 생성
           with st.spinner("🔄 결과 통합 및 응답 생성 중..."):
               st.caption("에이전트 결과를 통합하고 사용자 수준에 맞는 응답 생성")
               final_response = await self.universal_engine.integrate_and_adapt(
                   agent_results, meta_analysis, st.session_state.get('user_profile', {})
               )
           
           return final_response
   ```

4. WHEN showing agent collaboration THEN the system SHALL provide enhanced agent status visualization:
   ```python
   # A2A Agent 협업 상태 실시간 표시 (더 상세한 시각화)
   def render_agent_collaboration_status(self, selected_agents, workflow_status):
       if st.session_state.get('show_agent_details', True):
           with st.expander("🤖 A2A 에이전트 협업 상태", expanded=True):
               
               # 전체 워크플로우 진행률
               overall_progress = sum(agent.progress for agent in selected_agents) / len(selected_agents)
               st.progress(overall_progress / 100)
               st.caption(f"전체 진행률: {overall_progress:.1f}%")
               
               # 개별 에이전트 상태
               for agent in selected_agents:
                   col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                   
                   with col1:
                       st.write(f"**{agent.name}** (Port {agent.port})")
                       st.caption(f"역할: {agent.role}")
                   
                   with col2:
                       status_color = {
                           'running': '🟢 실행중',
                           'waiting': '🟡 대기중', 
                           'completed': '✅ 완료',
                           'error': '🔴 오류'
                       }
                       st.write(status_color.get(agent.status, '❓ 알 수 없음'))
                   
                   with col3:
                       st.write(f"{agent.progress}%")
                       st.progress(agent.progress / 100)
                   
                   with col4:
                       if agent.status == 'error':
                           if st.button("🔄", key=f"retry_{agent.id}"):
                               # 에이전트 재시작
                               await self.restart_agent(agent)
                       elif agent.status == 'completed':
                           if st.button("📊", key=f"details_{agent.id}"):
                               # 에이전트 결과 상세보기
                               st.session_state[f'show_agent_details_{agent.id}'] = True
               
               # 에이전트 간 데이터 흐름 시각화
               if len(selected_agents) > 1:
                   st.subheader("🔄 데이터 흐름")
                   workflow_diagram = self.generate_workflow_diagram(selected_agents, workflow_status)
                   st.graphviz_chart(workflow_diagram)
               
               # 실시간 로그
               if st.checkbox("실시간 로그 표시"):
                   log_container = st.container()
                   with log_container:
                       for log_entry in workflow_status.get('logs', [])[-10:]:  # 최근 10개
                           timestamp = log_entry['timestamp'].strftime("%H:%M:%S")
                           st.text(f"[{timestamp}] {log_entry['agent']}: {log_entry['message']}")
   ```

5. WHEN providing progressive disclosure THEN the system SHALL implement user-adaptive information revelation:
   ```python
   # 사용자 수준별 점진적 정보 공개 (더 정교한 적응)
   async def render_progressive_disclosure(self, analysis_result, user_profile):
       expertise_level = user_profile.get('expertise', 'auto_detect')
       
       if expertise_level == 'beginner' or (expertise_level == 'auto_detect' and 
                                           self.estimate_user_level() == 'beginner'):
           # 초보자용 친근한 설명
           st.write("😊 이 데이터를 보니 뭔가 공장에서 제품을 만드는 과정을 기록한 것 같네요.")
           st.write("마치 요리 레시피의 재료 분량을 측정한 기록처럼 보여요.")
           
           # 점진적 정보 공개 버튼들
           col1, col2, col3 = st.columns(3)
           with col1:
               if st.button("🔍 더 자세히 알아보기"):
                   st.session_state.disclosure_level = 'detailed'
           with col2:
               if st.button("📊 숫자로 보기"):
                   st.session_state.disclosure_level = 'numerical'
           with col3:
               if st.button("🎯 문제점 찾기"):
                   st.session_state.disclosure_level = 'problem_focused'
           
           # 선택된 공개 수준에 따른 추가 정보
           if st.session_state.get('disclosure_level') == 'detailed':
               st.write("좀 더 자세히 설명드리면...")
               st.write(analysis_result.get('detailed_explanation', ''))
               
               if st.button("🎓 전문가 수준으로 보기"):
                   st.session_state.user_profile['expertise'] = 'expert'
                   st.rerun()
       
       elif expertise_level == 'expert':
           # 전문가용 기술적 분석
           st.write("현재 Cpk 1.2에서 1.33으로 개선하려면 변동성을 약 8.3% 감소시켜야 합니다.")
           
           # 기술적 세부사항 즉시 표시
           with st.expander("📈 통계적 분석 결과", expanded=True):
               st.json(analysis_result.get('statistical_analysis', {}))
           
           with st.expander("🔧 개선 권장사항", expanded=True):
               for recommendation in analysis_result.get('expert_recommendations', []):
                   st.write(f"• {recommendation}")
           
           # 전문가용 추가 옵션
           col1, col2 = st.columns(2)
           with col1:
               if st.button("📊 고급 시각화"):
                   st.session_state.show_advanced_viz = True
           with col2:
               if st.button("🔬 심화 분석"):
                   st.session_state.request_deep_analysis = True
       
       else:  # intermediate or auto_detect
           # 중간 수준 설명
           st.write("데이터 분석 결과를 요약해드리겠습니다.")
           
           # 적응적 설명 깊이 조절
           explanation_depth = st.slider("설명 깊이", 1, 5, 3)
           
           if explanation_depth <= 2:
               st.write(analysis_result.get('simple_summary', ''))
           elif explanation_depth >= 4:
               st.write(analysis_result.get('detailed_analysis', ''))
           else:
               st.write(analysis_result.get('medium_summary', ''))
           
           # 사용자 반응에 따른 적응
           feedback = st.radio("이 설명이 도움이 되셨나요?", 
                             ["너무 어려워요", "적당해요", "더 자세히 알고 싶어요"])
           
           if feedback == "너무 어려워요":
               st.session_state.user_profile['expertise'] = 'beginner'
               st.info("💡 더 쉬운 설명으로 바꿔드리겠습니다.")
           elif feedback == "더 자세히 알고 싶어요":
               st.session_state.user_profile['expertise'] = 'expert'
               st.info("💡 더 전문적인 분석을 제공하겠습니다.")
   ```

6. WHEN handling user interactions THEN the system SHALL maintain session state and conversation flow:
   ```python
   # 세션 상태 관리 및 대화 흐름 유지 (더 포괄적인 컨텍스트 관리)
   def _get_session_context(self) -> Dict:
       """Universal Engine을 위한 포괄적 세션 컨텍스트"""
       return {
           # 사용자 프로필 및 학습 이력
           'user_profile': {
               'expertise': st.session_state.get('user_expertise', 'auto_detect'),
               'preferred_explanation_style': st.session_state.get('explanation_style', 'adaptive'),
               'domain_familiarity': st.session_state.get('domain_familiarity', {}),
               'learning_progress': st.session_state.get('learning_progress', {}),
               'interaction_patterns': st.session_state.get('interaction_patterns', [])
           },
           
           # 대화 이력 및 컨텍스트
           'conversation_history': st.session_state.get('messages', []),
           'conversation_topics': st.session_state.get('topics', []),
           'conversation_sentiment': st.session_state.get('sentiment_history', []),
           
           # 데이터 및 분석 이력
           'uploaded_files': st.session_state.get('uploaded_files', []),
           'current_data': st.session_state.get('current_data'),
           'data_context': st.session_state.get('data_context', {}),
           'previous_analyses': st.session_state.get('analysis_history', []),
           
           # A2A Agent 관련 컨텍스트
           'agent_preferences': st.session_state.get('agent_preferences', {}),
           'successful_agent_combinations': st.session_state.get('successful_combinations', []),
           'agent_performance_history': st.session_state.get('agent_performance', {}),
           
           # Universal Engine 메타 정보
           'meta_reasoning_history': st.session_state.get('meta_history', []),
           'reasoning_patterns': st.session_state.get('reasoning_patterns', {}),
           'adaptation_history': st.session_state.get('adaptation_history', []),
           
           # UI/UX 설정
           'ui_preferences': {
               'show_reasoning': st.session_state.get('show_reasoning', True),
               'show_agent_details': st.session_state.get('show_agent_details', True),
               'disclosure_level': st.session_state.get('disclosure_level', 'adaptive'),
               'visualization_preferences': st.session_state.get('viz_preferences', {})
           },
           
           # 성능 및 만족도 메트릭
           'performance_metrics': {
               'response_times': st.session_state.get('response_times', []),
               'satisfaction_scores': st.session_state.get('satisfaction_scores', []),
               'task_completion_rates': st.session_state.get('completion_rates', [])
           }
       }
   
   def update_session_context(self, interaction_result: Dict):
       """상호작용 결과를 바탕으로 세션 컨텍스트 업데이트"""
       
       # 사용자 프로필 업데이트
       if interaction_result.get('detected_expertise'):
           st.session_state.user_expertise = interaction_result['detected_expertise']
       
       # 대화 이력 업데이트
       if interaction_result.get('new_topics'):
           current_topics = st.session_state.get('topics', [])
           st.session_state.topics = current_topics + interaction_result['new_topics']
       
       # 메타 추론 이력 저장
       if interaction_result.get('meta_reasoning'):
           meta_history = st.session_state.get('meta_history', [])
           meta_history.append({
               'timestamp': datetime.now(),
               'reasoning': interaction_result['meta_reasoning'],
               'effectiveness': interaction_result.get('reasoning_effectiveness', 'unknown')
           })
           st.session_state.meta_history = meta_history[-50:]  # 최근 50개만 유지
       
       # 에이전트 성능 업데이트
       if interaction_result.get('agent_performance'):
           for agent_id, performance in interaction_result['agent_performance'].items():
               agent_perf = st.session_state.get('agent_performance', {})
               if agent_id not in agent_perf:
                   agent_perf[agent_id] = []
               agent_perf[agent_id].append(performance)
               st.session_state.agent_performance = agent_perf
   ```

7. WHEN displaying analysis results THEN the system SHALL provide enhanced result visualization:
   ```python
   def render_analysis_result(self, result: Dict[str, Any]):
       """Universal Engine 결과의 포괄적 시각화"""
       
       # 1. 메인 분석 결과 (사용자 수준에 맞게 적응)
       st.subheader("📊 분석 결과")
       
       user_level = st.session_state.get('user_expertise', 'auto_detect')
       if user_level == 'beginner':
           st.write(result.get('beginner_summary', ''))
           
           # 시각적 요소 강화
           if result.get('key_insights'):
               st.info("💡 주요 발견사항")
               for insight in result['key_insights'][:3]:  # 초보자는 3개만
                   st.write(f"• {insight}")
       
       elif user_level == 'expert':
           st.write(result.get('expert_analysis', ''))
           
           # 전문가용 상세 메트릭
           if result.get('detailed_metrics'):
               cols = st.columns(len(result['detailed_metrics']))
               for i, (metric, value) in enumerate(result['detailed_metrics'].items()):
                   with cols[i]:
                       st.metric(metric, value)
       
       # 2. Universal Engine 메타 분석 결과
       if result.get('meta_analysis') and st.session_state.get('show_reasoning', True):
           with st.expander("🧠 메타 추론 결과", expanded=False):
               meta = result['meta_analysis']
               
               col1, col2 = st.columns(2)
               with col1:
                   st.write("**감지된 컨텍스트:**")
                   st.json({
                       'domain': meta.get('domain', 'Unknown'),
                       'user_level': meta.get('user_level', 'Unknown'),
                       'intent': meta.get('intent', 'Unknown')
                   })
               
               with col2:
                   st.write("**추론 전략:**")
                   st.write(meta.get('reasoning_strategy', '전략 정보 없음'))
               
               # 추론 과정 시각화
               if meta.get('reasoning_steps'):
                   st.write("**추론 단계:**")
                   for i, step in enumerate(meta['reasoning_steps'], 1):
                       st.write(f"{i}. {step}")
       
       # 3. A2A Agent 기여도 및 협업 결과
       if result.get('agent_contributions'):
           with st.expander("🤖 에이전트 기여도", expanded=False):
               
               # 기여도 시각화
               agent_scores = {}
               for agent_id, contribution in result['agent_contributions'].items():
                   agent_scores[agent_id] = contribution.get('contribution_score', 0)
               
               if agent_scores:
                   # 기여도 차트
                   import plotly.express as px
                   fig = px.bar(
                       x=list(agent_scores.keys()), 
                       y=list(agent_scores.values()),
                       title="에이전트별 기여도"
                   )
                   st.plotly_chart(fig, use_container_width=True)
               
               # 개별 에이전트 결과
               for agent_id, contribution in result['agent_contributions'].items():
                   with st.container():
                       st.write(f"**🤖 {agent_id}**")
                       st.write(f"기여도: {contribution.get('contribution_score', 0):.1f}%")
                       st.write(f"요약: {contribution.get('summary', '작업 완료')}")
                       
                       if contribution.get('detailed_result'):
                           if st.button(f"상세 결과 보기", key=f"detail_{agent_id}"):
                               st.json(contribution['detailed_result'])
       
       # 4. 통합된 인사이트 및 권장사항
       if result.get('integrated_analysis'):
           st.subheader("🎯 통합 인사이트")
           st.write(result['integrated_analysis'])
       
       if result.get('recommendations'):
           st.subheader("💡 권장사항")
           for i, rec in enumerate(result['recommendations'], 1):
               st.write(f"{i}. {rec}")
       
       # 5. 대화형 후속 질문 제안
       if result.get('suggested_questions'):
           st.subheader("❓ 추가로 궁금한 점이 있으시다면")
           cols = st.columns(min(3, len(result['suggested_questions'])))
           
           for i, question in enumerate(result['suggested_questions'][:3]):
               with cols[i]:
                   if st.button(question, key=f"followup_{i}"):
                       # 후속 질문을 새로운 메시지로 추가
                       st.session_state.messages.append({
                           'role': 'user',
                           'content': question,
                           'timestamp': datetime.now(),
                           'source': 'suggested_followup'
                       })
                       st.rerun()
       
       # 6. 결과 품질 피드백
       st.subheader("📝 이 분석이 도움이 되셨나요?")
       col1, col2, col3, col4, col5 = st.columns(5)
       
       feedback_buttons = [
           ("😞", "매우 불만족", 1),
           ("😐", "불만족", 2), 
           ("😊", "보통", 3),
           ("😄", "만족", 4),
           ("🤩", "매우 만족", 5)
       ]
       
       for i, (emoji, label, score) in enumerate(feedback_buttons):
           with [col1, col2, col3, col4, col5][i]:
               if st.button(f"{emoji} {label}", key=f"feedback_{score}"):
                   # 피드백을 세션에 저장하고 Universal Engine 학습에 활용
                   self.record_user_feedback(result, score)
                   st.success("피드백이 저장되었습니다!")
   ```

8. WHEN providing recommendations THEN the system SHALL offer intelligent follow-up suggestions:
   ```python
   # 지능적 후속 추천 시스템 (더 정교한 추천 로직)
   async def generate_intelligent_followup_recommendations(self, analysis_result, user_profile, conversation_context):
       """Universal Engine 기반 지능적 후속 추천"""
       
       # Universal Engine으로 맥락적 추천 생성
       followup_recs = await self.universal_engine.generate_followup_recommendations(
           analysis_result=analysis_result,
           user_profile=user_profile,
           conversation_context=conversation_context,
           available_agents=self.available_agents
       )
       
       if followup_recs:
           st.subheader("💡 다음 단계 추천")
           
           # 추천 카테고리별 분류
           recommendation_categories = {
               'immediate': "🚀 즉시 실행 가능",
               'exploratory': "🔍 추가 탐색",
               'deep_dive': "🎯 심화 분석",
               'related': "🔗 관련 분석"
           }
           
           for category, title in recommendation_categories.items():
               category_recs = [r for r in followup_recs if r.get('category') == category]
               
               if category_recs:
                   with st.expander(title, expanded=(category == 'immediate')):
                       cols = st.columns(min(2, len(category_recs)))
                       
                       for i, rec in enumerate(category_recs):
                           with cols[i % 2]:
                               # 추천 카드 스타일
                               with st.container():
                                   st.write(f"**{rec['title']}**")
                                   st.caption(rec.get('description', ''))
                                   
                                   # 예상 소요 시간 및 복잡도 표시
                                   col_time, col_complexity = st.columns(2)
                                   with col_time:
                                       st.caption(f"⏱️ {rec.get('estimated_time', '알 수 없음')}")
                                   with col_complexity:
                                       complexity_icons = {
                                           'easy': '🟢 쉬움',
                                           'medium': '🟡 보통', 
                                           'hard': '🔴 어려움'
                                       }
                                       st.caption(complexity_icons.get(rec.get('complexity', 'medium')))
                                   
                                   # 추천 실행 버튼
                                   if st.button(f"▶️ 실행", key=f"rec_{rec['id']}"):
                                       # 추천 클릭 시 Universal Engine으로 처리
                                       new_message = {
                                           'role': 'user',
                                           'content': rec['query'],
                                           'timestamp': datetime.now(),
                                           'source': 'recommendation',
                                           'recommendation_id': rec['id']
                                       }
                                       st.session_state.messages.append(new_message)
                                       
                                       # 추천 클릭 이벤트 기록 (학습용)
                                       self.record_recommendation_click(rec, user_profile)
                                       st.rerun()
           
           # 사용자 맞춤 추천 학습
           st.subheader("🎯 맞춤 추천 개선")
           col1, col2 = st.columns(2)
           
           with col1:
               if st.button("👍 이런 추천이 좋아요"):
                   self.learn_recommendation_preferences(followup_recs, 'positive')
                   st.success("추천 선호도가 학습되었습니다!")
           
           with col2:
               if st.button("👎 다른 종류 추천 원해요"):
                   self.learn_recommendation_preferences(followup_recs, 'negative')
                   st.info("추천 방식을 조정하겠습니다!")
   ```

9. WHEN handling errors THEN the system SHALL provide user-friendly error messages and recovery options:
   ```python
   # 사용자 친화적 오류 처리 (더 포괄적인 오류 처리 및 복구)
   async def handle_analysis_errors(self, user_query: str):
       """Universal Engine + A2A 통합 시스템의 포괄적 오류 처리"""
       
       try:
           result = await self.universal_a2a_system.process_unified_query(
               query=user_query,
               data=st.session_state.get('current_data'),
               context=self._get_session_context()
           )
           return result
           
       except A2AAgentError as e:
           # A2A Agent 관련 오류
           st.error(f"🤖 에이전트 오류: {e.agent_id}가 일시적으로 사용할 수 없습니다.")
           
           # 구체적 오류 유형별 처리
           if e.error_type == 'connection_timeout':
               st.info("💡 네트워크 연결 문제로 보입니다. 잠시 후 다시 시도해주세요.")
           elif e.error_type == 'agent_overload':
               st.info("💡 에이전트가 과부하 상태입니다. 다른 에이전트로 분석을 계속하시겠습니까?")
           elif e.error_type == 'data_format_error':
               st.info("💡 데이터 형식에 문제가 있습니다. 데이터를 다시 확인해주세요.")
           
           # 복구 옵션 제공
           col1, col2, col3 = st.columns(3)
           
           with col1:
               if st.button("🔄 재시도"):
                   st.rerun()
           
           with col2:
               if st.button("🤖 다른 에이전트 사용"):
                   # 문제가 있는 에이전트 제외하고 재시도
                   st.session_state.excluded_agents = st.session_state.get('excluded_agents', [])
                   st.session_state.excluded_agents.append(e.agent_id)
                   st.rerun()
           
           with col3:
               if st.button("📞 지원 요청"):
                   st.session_state.show_support_form = True
           
           # 대안 분석 제안
           st.subheader("🔄 대안 분석 방법")
           alternative_approaches = await self.universal_engine.suggest_alternative_approaches(
               original_query=user_query,
               failed_agent=e.agent_id,
               available_agents=[a for a in self.available_agents if a.id != e.agent_id]
           )
           
           for approach in alternative_approaches:
               if st.button(f"💡 {approach['title']}", key=f"alt_{approach['id']}"):
                   st.session_state.messages.append({
                       'role': 'user',
                       'content': approach['modified_query'],
                       'timestamp': datetime.now(),
                       'source': 'error_recovery'
                   })
                   st.rerun()
       
       except UniversalEngineError as e:
           # Universal Engine 관련 오류
           st.error(f"🧠 분석 엔진 오류: {str(e)}")
           
           # 오류 유형별 사용자 친화적 메시지
           if e.error_type == 'context_analysis_failed':
               st.info("💡 데이터 컨텍스트 분석에 실패했습니다. 데이터에 대해 더 구체적으로 설명해주시면 도움이 됩니다.")
               
               # 컨텍스트 명확화 도움
               st.subheader("📝 데이터 정보 입력")
               data_description = st.text_area(
                   "데이터에 대해 설명해주세요:",
                   placeholder="예: 반도체 공정 데이터, 매출 데이터, 고객 설문조사 등"
               )
               
               if data_description and st.button("🔄 다시 분석"):
                   st.session_state.user_provided_context = data_description
                   st.rerun()
           
           elif e.error_type == 'meta_reasoning_failed':
               st.info("💡 메타 추론에 실패했습니다. 질문을 더 구체적으로 바꿔서 다시 시도해보세요.")
               
               # 질문 개선 도움
               st.subheader("❓ 질문 개선 도움")
               question_suggestions = [
                   "이 데이터에서 어떤 패턴을 찾을 수 있나요?",
                   "데이터에 문제가 있는 부분이 있나요?",
                   "이 결과를 어떻게 해석해야 하나요?",
                   "다음에 무엇을 해야 하나요?"
               ]
               
               for suggestion in question_suggestions:
                   if st.button(f"💡 {suggestion}", key=f"suggest_{hash(suggestion)}"):
                       st.session_state.messages.append({
                           'role': 'user',
                           'content': suggestion,
                           'timestamp': datetime.now(),
                           'source': 'error_recovery_suggestion'
                       })
                       st.rerun()
           
           elif e.error_type == 'user_level_detection_failed':
               st.info("💡 사용자 수준 감지에 실패했습니다. 수동으로 설정해주세요.")
               
               # 수동 사용자 수준 설정
               manual_level = st.selectbox(
                   "전문성 수준을 선택해주세요:",
                   ["초보자 - 쉬운 설명 원함", "중급자 - 적당한 설명", "전문가 - 기술적 분석 원함"]
               )
               
               if st.button("✅ 설정 완료"):
                   level_mapping = {
                       "초보자 - 쉬운 설명 원함": "beginner",
                       "중급자 - 적당한 설명": "intermediate", 
                       "전문가 - 기술적 분석 원함": "expert"
                   }
                   st.session_state.user_expertise = level_mapping[manual_level]
                   st.success("사용자 수준이 설정되었습니다!")
                   st.rerun()
       
       except DataProcessingError as e:
           # 데이터 처리 오류
           st.error(f"📊 데이터 처리 오류: {str(e)}")
           
           # 데이터 문제 진단 및 해결책 제안
           st.subheader("🔍 데이터 문제 진단")
           
           diagnostic_results = await self.diagnose_data_issues(st.session_state.get('current_data'))
           
           for issue in diagnostic_results:
               st.warning(f"⚠️ {issue['problem']}")
               st.info(f"💡 해결책: {issue['solution']}")
               
               if issue.get('auto_fix_available'):
                   if st.button(f"🔧 자동 수정", key=f"fix_{issue['id']}"):
                       fixed_data = await self.auto_fix_data_issue(issue)
                       st.session_state.current_data = fixed_data
                       st.success("데이터가 수정되었습니다!")
                       st.rerun()
       
       except Exception as e:
           # 예상치 못한 오류
           st.error("😵 예상치 못한 오류가 발생했습니다.")
           
           # 오류 보고 및 복구
           with st.expander("🔧 오류 세부 정보", expanded=False):
               st.code(str(e))
           
           st.info("💡 이 오류를 개발팀에 보고하여 시스템을 개선하는데 도움을 주세요.")
           
           col1, col2 = st.columns(2)
           with col1:
               if st.button("📧 오류 보고"):
                   self.send_error_report(e, user_query, st.session_state.get('current_data'))
                   st.success("오류가 보고되었습니다!")
           
           with col2:
               if st.button("🏠 홈으로 돌아가기"):
                   # 세션 초기화하고 홈으로
                   for key in list(st.session_state.keys()):
                       if key not in ['user_profile', 'user_expertise']:  # 사용자 설정은 유지
                           del st.session_state[key]
                   st.rerun()
   ```

10. WHEN system loads THEN the system SHALL initialize all components seamlessly:
    ```python
    async def initialize(self):
        """Universal Engine + A2A 통합 시스템의 완전한 초기화"""
        
        # 초기화 진행 상태 표시
        initialization_container = st.container()
        
        with initialization_container:
            st.info("🍒 Cherry AI Universal Engine 초기화 중...")
            
            # 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1단계: Universal Engine 초기화 (20%)
                status_text.text("🧠 Universal Engine 초기화 중...")
                self.universal_engine = UniversalEngine()
                await self.universal_engine.initialize()
                progress_bar.progress(20)
                
                # 2단계: A2A Agent 발견 (40%)
                status_text.text("🤖 A2A 에이전트 발견 중...")
                self.available_agents = await self.discover_a2a_agents()
                progress_bar.progress(40)
                
                # 3단계: Universal Engine + A2A 통합 시스템 초기화 (60%)
                status_text.text("🔗 통합 시스템 초기화 중...")
                self.universal_a2a_system = UniversalEngineA2AIntegration(
                    universal_engine=self.universal_engine,
                    available_agents=self.available_agents
                )
                await self.universal_a2a_system.initialize()
                progress_bar.progress(60)
                
                # 4단계: Agent 상태 확인 (80%)
                status_text.text("✅ 에이전트 상태 확인 중...")
                active_agents = []
                for agent in self.available_agents:
                    try:
                        status = await self.check_agent_status(agent)
                        if status == 'active':
                            active_agents.append(agent)
                    except Exception as e:
                        st.warning(f"⚠️ {agent.name} (포트 {agent.port}) 연결 실패: {str(e)}")
                
                self.available_agents = active_agents
                progress_bar.progress(80)
                
                # 5단계: UI 컴포넌트 초기화 (100%)
                status_text.text("🎨 UI 컴포넌트 초기화 중...")
                await self.initialize_ui_components()
                progress_bar.progress(100)
                
                # 초기화 완료 메시지
                status_text.empty()
                progress_bar.empty()
                
                if len(self.available_agents) > 0:
                    st.success(f"✅ Cherry AI Universal Engine 초기화 완료!")
                    st.info(f"🤖 {len(self.available_agents)}개 A2A 에이전트와 연결됨")
                    
                    # 연결된 에이전트 목록 표시
                    with st.expander("🤖 연결된 에이전트 목록", expanded=False):
                        for agent in self.available_agents:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{agent.name}**")
                            with col2:
                                st.write(f"포트: {agent.port}")
                            with col3:
                                st.write("🟢 활성")
                else:
                    st.warning("⚠️ A2A 에이전트 연결 실패 - 기본 Universal Engine 기능만 사용 가능")
                    st.info("💡 에이전트 서버를 시작한 후 페이지를 새로고침해주세요.")
                
                # 초기화 성공 시 환영 메시지
                if not st.session_state.get('welcome_shown', False):
                    self.show_welcome_message()
                    st.session_state.welcome_shown = True
                
                return True
                
            except Exception as e:
                # 초기화 실패 처리
                progress_bar.empty()
                status_text.empty()
                
                st.error(f"❌ 초기화 실패: {str(e)}")
                
                # 부분 초기화 옵션 제공
                st.subheader("🔄 복구 옵션")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 다시 시도"):
                        st.rerun()
                
                with col2:
                    if st.button("🧠 Universal Engine만 사용"):
                        # A2A 없이 Universal Engine만 초기화
                        try:
                            self.universal_engine = UniversalEngine()
                            await self.universal_engine.initialize()
                            self.available_agents = []
                            st.success("✅ Universal Engine 기본 모드로 시작됨")
                            return True
                        except Exception as e2:
                            st.error(f"기본 모드 초기화도 실패: {str(e2)}")
                
                with col3:
                    if st.button("📞 지원 요청"):
                        st.session_state.show_support_form = True
                
                return False
    
    def show_welcome_message(self):
        """사용자 환영 메시지 및 시작 가이드"""
        st.balloons()  # 축하 효과
        
        st.success("🎉 Cherry AI Universal Engine에 오신 것을 환영합니다!")
        
        with st.expander("🚀 시작하기 가이드", expanded=True):
            st.markdown("""
            ### 🍒 Cherry AI Universal Engine 특징
            
            - **🧠 LLM First 접근**: 하드코딩 없는 진정한 지능형 분석
            - **🤖 A2A Agent 통합**: 10개 전문 에이전트와 자동 협업
            - **🎯 사용자 적응**: 초보자부터 전문가까지 자동 수준 조절
            - **🔍 메타 추론**: 생각에 대해 생각하는 고급 추론
            - **📊 범용 분석**: 모든 도메인, 모든 데이터 유형 지원
            
            ### 📝 사용 방법
            
            1. **📁 데이터 업로드**: 왼쪽 사이드바에서 파일 업로드
            2. **💬 자연어 질문**: "이 데이터가 뭘 말하는지 모르겠어요" 같은 자연스러운 질문
            3. **🔍 점진적 탐색**: 시스템이 제안하는 후속 질문으로 깊이 있는 분석
            4. **⚙️ 개인화**: 시스템이 자동으로 당신의 수준에 맞춰 설명 조절
            
            ### 💡 시작 예시
            """)
            
            # 예시 질문 버튼들
            example_questions = [
                "이 데이터 파일이 뭘 말하는지 전혀 모르겠어요. 도움 주세요.",
                "데이터에서 이상한 패턴이나 문제점을 찾아주세요.",
                "이 결과를 어떻게 해석하고 다음에 뭘 해야 하나요?",
                "전문가 수준의 상세한 통계 분석을 원합니다."
            ]
            
            st.write("**🎯 예시 질문 (클릭하면 바로 시작):**")
            
            for i, question in enumerate(example_questions):
                if st.button(f"💡 {question}", key=f"example_{i}"):
                    st.session_state.messages.append({
                        'role': 'user',
                        'content': question,
                        'timestamp': datetime.now(),
                        'source': 'welcome_example'
                    })
                    st.rerun()
    ```

### Requirement 11: Performance and Scalability

**User Story:** As a system administrator, I want the universal engine to perform efficiently at scale while maintaining response quality, so that it can serve multiple users simultaneously without degradation.

#### Acceptance Criteria

1. WHEN processing queries THEN the system SHALL maintain response times under 10 seconds for typical analysis requests
2. WHEN multiple users access simultaneously THEN the system SHALL handle concurrent requests without performance degradation
3. WHEN learning from interactions THEN the system SHALL update knowledge efficiently without blocking other operations
4. WHEN system load increases THEN the system SHALL gracefully scale processing capacity
5. WHEN monitoring performance THEN the system SHALL provide metrics on response time, accuracy, user satisfaction, and resource utilization
### 
Requirement 12: Semantic Routing & Intent Recognition (2025 연구 기반)

**User Story:** As a user, I want the system to understand my intent through semantic analysis rather than keyword matching, so that I get relevant responses even when my questions are ambiguous or use non-standard terminology.

#### Acceptance Criteria

1. WHEN processing queries THEN the system SHALL implement Universal Intent Detection using the exact pattern:
   ```
   # 범용 의도 분석 패턴
   사전 정의된 카테고리나 패턴에 의존하지 않고
   쿼리 자체가 말하는 것을 들어보겠습니다:

   1. 직접적 의도 분석:
      - 사용자가 명시적으로 요청한 것은?
      - 사용된 언어의 톤과 스타일은?
      - 기대하는 응답의 형태는?

   2. 암묵적 의도 추론:
      - 표면적 질문 뒤의 진정한 관심사는?
      - 현재 상황이나 문제 상황의 맥락은?
      - 궁극적으로 해결하고자 하는 것은?

   3. 동적 분류:
      - 이 쿼리는 어떤 종류의 도움이 필요한가?
      - 탐색적 분석? 문제 해결? 학습? 검증?
      - 즉시 답변? 단계별 가이드? 심화 분석?

   카테고리에 맞추려 하지 말고, 
   쿼리가 자연스럽게 이끄는 방향을 따르겠습니다.
   ```

2. WHEN exploring semantic space THEN the system SHALL use Semantic Space Navigation pattern:
   ```
   # 의미 공간 탐색 패턴
   이 쿼리와 데이터가 위치한 의미 공간을 탐색해보겠습니다:

   1. 의미적 근접성 분석:
      - 어떤 개념들이 연관되어 있는가?
      - 다른 유사한 상황들은 어떻게 처리되었는가?
      - 관련 도메인 지식은 무엇인가?

   2. 맥락적 연결 탐색:
      - 이 문제와 관련된 다른 측면들은?
      - 상위 개념이나 하위 세부사항들은?
      - 인과관계나 상관관계는?

   3. 동적 지식 연결:
      - 실시간으로 관련 지식 검색
      - 다양한 관점에서의 접근법 고려
      - 최신 연구나 모범 사례 통합

   의미 공간에서 자연스럽게 형성되는 연결을 따라
   최적의 분석 경로를 찾겠습니다.
   ```

3. WHEN intent is ambiguous THEN the system SHALL distinguish between direct intent (명시적 요청) and implicit intent (암묵적 의도)
4. WHEN multiple interpretations exist THEN the system SHALL explore all semantic possibilities before selecting the most relevant approach
5. WHEN intent detection is complete THEN the system SHALL select response strategy based on discovered intent rather than predefined templates

### Requirement 13: Chain-of-Thought with Self-Consistency

**User Story:** As a user, I want the system to use multiple reasoning paths and validate consistency across them, so that I can trust the reliability of complex analysis results.

#### Acceptance Criteria

1. WHEN analyzing complex problems THEN the system SHALL use the exact Chain-of-Thought pattern:
   ```
   # 다중 추론 경로 패턴
   이 문제를 여러 관점에서 분석해보겠습니다:

   추론 경로 1: 데이터 중심 접근
   - 데이터가 보여주는 패턴은?
   - 통계적으로 유의미한 특징은?
   - 데이터만으로 도출할 수 있는 결론은?

   추론 경로 2: 도메인 지식 중심 접근  
   - 이 분야의 일반적인 원리는?
   - 전문가들이 주로 사용하는 방법은?
   - 업계 모범 사례는?

   추론 경로 3: 사용자 맥락 중심 접근
   - 사용자의 상황과 제약사항은?
   - 실제 적용 가능성은?
   - 우선순위와 목표는?

   일관성 검증:
   - 각 추론 경로의 결론이 일치하는가?
   - 차이가 있다면 그 이유는?
   - 가장 신뢰할 만한 결론은?

   최종적으로 가장 일관성 있고 신뢰할 만한 분석을 제시하겠습니다.
   ```

2. WHEN reasoning paths conflict THEN the system SHALL explicitly identify and explain the differences
3. WHEN consistency is achieved THEN the system SHALL present the validated conclusion with confidence level
4. WHEN consistency cannot be achieved THEN the system SHALL present multiple valid interpretations with their respective strengths
5. WHEN final analysis is complete THEN the system SHALL state confidence level and areas of remaining uncertainty

### Requirement 14: Zero-Shot Adaptive Reasoning Without Templates

**User Story:** As a user with novel or unique problems, I want the system to reason from first principles without relying on templates, so that I get fresh insights tailored to my specific situation.

#### Acceptance Criteria

1. WHEN encountering novel problems THEN the system SHALL use the exact Zero-Shot Adaptive Reasoning pattern:
   ```
   # 무작정 적응적 추론 패턴
   이전 사례나 템플릿 없이 순수하게 추론해보겠습니다:

   1. 문제 공간 정의:
      - 이 문제의 본질은 무엇인가?
      - 어떤 종류의 추론이 필요한가?
      - 해결해야 할 핵심 질문들은?

   2. 추론 전략 수립:
      - 어떤 순서로 접근할 것인가?
      - 어떤 정보가 추가로 필요한가?
      - 어떤 방법론이 가장 적합한가?

   3. 단계별 추론 실행:
      - 각 단계에서 논리적 타당성 확인
      - 가정과 제약사항 명시
      - 불확실성과 신뢰도 평가

   4. 결과 통합 및 검증:
      - 부분 결론들이 전체적으로 일관성 있는가?
      - 대안적 해석이나 설명이 가능한가?
      - 결론의 한계와 적용 범위는?

   템플릿이나 공식에 의존하지 않고
   문제 자체의 본질에 맞는 추론을 수행하겠습니다.
   ```

2. WHEN making assumptions THEN the system SHALL explicitly state "가정과 제약사항 명시"
3. WHEN evaluating uncertainty THEN the system SHALL provide "불확실성과 신뢰도 평가"
4. WHEN reasoning is complete THEN the system SHALL validate that no templates or formulas were used
5. WHEN presenting results THEN the system SHALL clearly indicate the reasoning approach was tailored to the specific problem

### Requirement 15: 실제 작동 시나리오 구현

**User Story:** As a user, I want the system to handle real-world scenarios with the same quality as demonstrated in the specification examples, so that I can rely on consistent performance across different use cases.

#### Acceptance Criteria

1. WHEN user is complete beginner THEN the system SHALL handle the exact scenario:
   ```
   입력: "이 데이터 파일이 뭘 말하는지 전혀 모르겠어요. 도움 주세요."
   데이터: [복잡한 반도체 공정 데이터]
   
   응답 패턴:
   안녕하세요! 😊 
   
   이 데이터를 보니 뭔가 공장에서 제품을 만드는 과정을 기록한 것 같네요. 
   마치 요리 레시피의 재료 분량을 측정한 기록처럼 보여요.
   
   일단 몇 가지 흥미로운 패턴이 보이는데요:
   1. 숫자들이 일정한 범위 안에서 움직이고 있어요
   2. 시간에 따라 변화하는 모습이 있어요
   3. 몇 개의 주요 측정값들이 있는 것 같아요
   
   어떤 부분이 가장 궁금하세요? 
   - 이 숫자들이 무엇을 의미하는지?
   - 좋은 건지 나쁜 건지?
   - 뭔가 문제가 있는 건지?
   
   하나씩 천천히 알아가봐요! 🔍
   ```

2. WHEN user is expert THEN the system SHALL handle the exact scenario:
   ```
   입력: "공정 능력 지수가 1.2인데 타겟을 1.33으로 올리려면 어떤 파라미터를 조정해야 할까? 특히 도즈 균일성 측면에서 분석 부탁해."
   
   응답 패턴:
   현재 Cpk 1.2에서 1.33으로 개선하려면 변동성을 약 8.3% 감소시켜야 합니다.
   
   ## 도즈 균일성 분석 결과
   
   **현재 상태:**
   - 도즈 균일성: ±1.8% (3σ)
   - 주요 변동 요인: 웨이퍼 중심부 과도즈 (1.2% 편차)
   - 에지 영역 도즈 부족: 평균 대비 -2.1%
   
   **Cpk 1.33 달성을 위한 핵심 파라미터 조정:**
   
   1. **빔 스캔 최적화** (예상 개선: 40%)
      - 스캔 속도: 현재 대비 5-7% 감소
      - 스캔 패턴: Raster → Serpentine 변경 검토
      - 예상 균일성 개선: ±1.8% → ±1.4%
   
   [상세 기술 분석 계속...]
   ```

3. WHEN user has ambiguous question THEN the system SHALL handle the exact scenario:
   ```
   입력: "뭔가 이상한데요? 평소랑 다른 것 같아요."
   
   응답 패턴:
   네, 뭔가 평소와 다르다는 느낌이 드시는군요. 함께 살펴보겠습니다! 🔍
   
   일단 데이터를 빠르게 스캔해보니 몇 가지 눈에 띄는 부분이 있어요:
   
   **즉시 발견된 이상 징후:**
   1. 📈 측정값 #3이 평소보다 15% 높음 (마지막 2시간)
   2. 📊 변동폭이 평소의 2배로 증가 (오후 2시 이후)
   3. ⚠️ 센서 #7에서 간헐적 스파이크 발생
   
   **명확화가 필요한 부분:**
   - "평소"라고 하신 것은 언제와 비교하신 건가요?
   - 특별히 어떤 부분이 이상하게 느껴지셨나요?
   
   일단 가장 눈에 띄는 것부터 보여드릴까요?
   ```

4. WHEN adapting to user responses THEN the system SHALL dynamically adjust explanation depth and technical language
5. WHEN scenarios evolve THEN the system SHALL maintain consistency with the demonstrated interaction patterns

### Requirement 16: Dynamic Knowledge Orchestrator Implementation

**User Story:** As a user, I want the system to dynamically integrate knowledge from multiple sources and reasoning approaches, so that I get comprehensive analysis that considers all relevant aspects.

#### Acceptance Criteria

1. WHEN implementing knowledge orchestration THEN the system SHALL use the exact architecture:
   ```python
   class DynamicKnowledgeOrchestrator:
       """동적 지식 오케스트레이터 - 실시간 지식 통합"""
       
       async def retrieve_relevant_knowledge(self, context: Dict):
           """컨텍스트에 맞는 지식 실시간 검색"""
           
       async def reason_with_context(self, knowledge: Dict, query: str):
           """맥락을 고려한 추론 수행"""
           
       async def collaborate_with_agents(self, reasoning_result: Dict):
           """다중 에이전트와 협업"""
           
       async def self_reflect_and_refine(self, result: Dict):
           """결과를 자가 검토하고 개선"""
   ```

2. WHEN retrieving knowledge THEN the system SHALL search contextually relevant information without predefined categories
3. WHEN reasoning with context THEN the system SHALL integrate domain knowledge, user context, and data patterns
4. WHEN collaborating with agents THEN the system SHALL coordinate multiple analysis approaches
5. WHEN refining results THEN the system SHALL apply self-reflection to improve analysis quality

### Requirement 17: Adaptive Response Generator with Progressive Disclosure

**User Story:** As a user, I want responses that adapt to my understanding level and provide information progressively, so that I can learn and explore at my own pace.

#### Acceptance Criteria

1. WHEN implementing response generation THEN the system SHALL use the exact architecture:
   ```python
   class AdaptiveResponseGenerator:
       """적응형 응답 생성기 - 사용자 맞춤 응답"""
       
       async def generate_expertise_aware_explanation(self, analysis: Dict, user_level: str):
           """사용자 수준에 맞는 설명 생성"""
           
       async def progressive_disclosure(self, information: Dict, user_response: str):
           """점진적 정보 공개"""
           
       async def interactive_clarification(self, uncertainty: Dict):
           """대화형 명확화 질문"""
           
       async def recommend_followup(self, current_analysis: Dict, user_interest: Dict):
           """후속 분석 추천"""
   ```

2. WHEN generating explanations THEN the system SHALL adapt language complexity based on estimated user expertise
3. WHEN providing progressive disclosure THEN the system SHALL reveal information based on user interest and comprehension
4. WHEN uncertainty exists THEN the system SHALL ask interactive clarification questions
5. WHEN analysis is complete THEN the system SHALL recommend relevant follow-up analyses or explorations

### Requirement 18: Real-time Learning System Implementation

**User Story:** As a user, I want the system to learn from our interactions and continuously improve, so that the analysis quality gets better over time.

#### Acceptance Criteria

1. WHEN implementing learning system THEN the system SHALL use the exact pattern:
   ```python
   class RealTimeLearningSystem:
       def __init__(self):
           self.user_feedback_history = []
           self.successful_patterns = {}
           self.failure_patterns = {}
           
       async def learn_from_interaction(self, interaction: Dict):
           learning_prompt = f"""
           이번 상호작용에서 배운 것을 정리하겠습니다:
           
           상호작용: {interaction}
           사용자 만족도: {interaction.get('satisfaction', 'unknown')}
           
           학습 포인트:
           1. 성공한 부분: 무엇이 효과적이었는가?
           2. 개선 필요: 무엇이 부족했는가?
           3. 일반화 가능: 다른 상황에도 적용할 수 있는 패턴은?
           4. 주의사항: 피해야 할 접근법은?
           
           이 학습을 향후 유사한 상황에서 활용하겠습니다.
           """
   ```

2. WHEN user provides feedback THEN the system SHALL analyze and incorporate learning points
3. WHEN successful patterns are identified THEN the system SHALL generalize for future similar situations
4. WHEN failures occur THEN the system SHALL identify and avoid similar approaches
5. WHEN knowledge is updated THEN the system SHALL maintain privacy while preserving learning

### Requirement 19: Performance Metrics and Validation

**User Story:** As a system administrator, I want comprehensive metrics to validate the system's performance against the specification goals, so that I can ensure it meets the LLM First Universal Engine requirements.

#### Acceptance Criteria

1. WHEN measuring user satisfaction THEN the system SHALL track the exact metrics:
   - **응답 적절성**: 사용자 피드백 기반 1-5점 평가
   - **이해도 향상**: 후속 질문 감소율
   - **문제 해결률**: 사용자 목표 달성 비율
   - **재사용률**: 동일 사용자의 반복 사용 빈도

2. WHEN measuring system performance THEN the system SHALL track:
   - **적응 속도**: 사용자 수준 파악에 필요한 상호작용 수
   - **정확도**: 도메인 감지 및 의도 분석 정확도
   - **효율성**: 평균 응답 시간 및 처리 속도
   - **확장성**: 새로운 도메인 적응 속도

3. WHEN conducting A/B testing THEN the system SHALL compare against hardcoded baseline systems
4. WHEN validating scenarios THEN the system SHALL test all specification examples exactly as documented
5. WHEN reporting metrics THEN the system SHALL provide comprehensive performance dashboards

### Requirement 20: A2A Agent Integration and Dynamic Orchestration

**User Story:** As a system user, I want the Universal Engine to intelligently coordinate with existing A2A agents to leverage their specialized capabilities, so that I get comprehensive analysis without losing the benefits of specialized tools.

#### Acceptance Criteria

1. WHEN processing queries THEN the system SHALL dynamically discover and integrate with available A2A agents:
   - **Data Cleaning Server (Port 8306)**: 🧹 LLM 기반 지능형 데이터 정리, 빈 데이터 처리, 7단계 표준 정리 프로세스
   - **Data Loader Server (Port 8307)**: 📁 통합 데이터 로딩, UTF-8 인코딩 문제 해결, 다양한 파일 형식 지원
   - **Data Visualization Server (Port 8308)**: 📊 Interactive 시각화, Plotly 기반 차트 생성
   - **Data Wrangling Server (Port 8309)**: 🔧 데이터 변환, 조작, 구조 변경
   - **Feature Engineering Server (Port 8310)**: ⚙️ 피처 생성, 변환, 선택, 차원 축소
   - **SQL Database Server (Port 8311)**: 🗄️ SQL 쿼리 실행, 데이터베이스 연결
   - **EDA Tools Server (Port 8312)**: 🔍 탐색적 데이터 분석, 통계 계산, 패턴 발견
   - **H2O ML Server (Port 8313)**: 🤖 머신러닝 모델링, AutoML, 예측 분석
   - **MLflow Tools Server (Port 8314)**: 📈 모델 관리, 실험 추적, 버전 관리
   - **Pandas Collaboration Hub (Port 8315)**: 🐼 판다스 기반 데이터 조작 및 분석

2. WHEN selecting A2A agents THEN the system SHALL use LLM-based dynamic agent selection without hardcoded rules:
   ```python
   # 제거해야 할 하드코딩 패턴:
   if "clean" in query:
       use_data_cleaning_agent()
   elif "visualize" in query:
       use_visualization_agent()
   
   # 대신 사용할 LLM 기반 동적 선택:
   agent_selection_prompt = f"""
   사용자 요청: {query}
   데이터 특성: {data_characteristics}
   
   다음 A2A 에이전트들 중 이 요청을 처리하기 위해 필요한 에이전트들을 선택하세요:
   - data_cleaning: 데이터 정제, 결측치 처리, 이상치 제거
   - data_loader: 파일 로딩, 데이터 파싱, 인코딩 처리
   - eda_tools: 탐색적 분석, 기초 통계, 패턴 발견
   [... 모든 에이전트 설명]
   
   요청의 본질을 파악하여 최적의 에이전트 조합과 실행 순서를 결정하세요.
   """
   ```

3. WHEN coordinating A2A agents THEN the system SHALL implement intelligent workflow orchestration:
   ```python
   class A2AWorkflowOrchestrator:
       async def execute_agent_workflow(self, selected_agents: List, query: str, data: Any) -> Dict:
           """
           A2A 에이전트 워크플로우 동적 실행
           - 순차 실행: data_loader → data_cleaning → eda_tools → feature_engineering → h2o_ml
           - 병렬 실행: visualization + sql_database (독립적 분석)
           - 결과 통합: pandas_collaboration_hub가 최종 통합
           """
   ```

4. WHEN integrating with A2A agents THEN the system SHALL maintain A2A SDK 0.2.9 standard compliance:
   - TaskUpdater 실시간 스트리밍 지원
   - AgentCard, AgentSkill, AgentCapabilities 표준 준수
   - RequestContext 및 EventQueue 활용
   - UnifiedDataInterface 패턴 사용

5. WHEN A2A agents return results THEN the system SHALL synthesize results using Universal Engine meta-reasoning:
   ```python
   synthesis_prompt = f"""
   A2A 에이전트 실행 결과들을 통합 분석하겠습니다:
   
   Data Cleaning 결과: {cleaning_result}
   EDA Tools 결과: {eda_result}
   Visualization 결과: {viz_result}
   ML 결과: {ml_result}
   
   사용자 프로필: {user_profile}
   원본 쿼리: {original_query}
   
   각 에이전트의 결과를 종합하여:
   1. 핵심 인사이트 추출
   2. 사용자 수준에 맞는 설명 생성
   3. 실행 가능한 다음 단계 제안
   4. 결과 간 일관성 검증
   
   통합된 분석 결과를 제시하세요.
   """
   ```

### Requirement 21: A2A Agent Discovery and Health Monitoring

**User Story:** As a system administrator, I want the Universal Engine to automatically discover available A2A agents and monitor their health, so that the system can adapt to agent availability changes without manual configuration.

#### Acceptance Criteria

1. WHEN system starts THEN it SHALL automatically discover available A2A agents by checking standard ports:
   ```python
   AGENT_PORTS = {
       "data_cleaning": 8306,
       "data_loader": 8307,
       "data_visualization": 8308,
       "data_wrangling": 8309,
       "feature_engineering": 8310,
       "sql_database": 8311,
       "eda_tools": 8312,
       "h2o_ml": 8313,
       "mlflow_tools": 8314,
       "pandas_collaboration_hub": 8315
   }
   ```

2. WHEN discovering agents THEN the system SHALL validate A2A agent capabilities through /.well-known/agent.json endpoint
3. WHEN agents become unavailable THEN the system SHALL gracefully adapt workflow without those agents
4. WHEN agents return errors THEN the system SHALL implement intelligent fallback strategies
5. WHEN monitoring agent health THEN the system SHALL track response times, success rates, and availability metrics

### Requirement 22: A2A Agent Result Integration and Quality Assurance

**User Story:** As a user, I want the system to intelligently combine results from multiple A2A agents into coherent insights, so that I get comprehensive analysis rather than fragmented outputs.

#### Acceptance Criteria

1. WHEN multiple A2A agents execute THEN the system SHALL validate result consistency across agents
2. WHEN agent results conflict THEN the system SHALL use meta-reasoning to resolve conflicts:
   ```python
   conflict_resolution_prompt = f"""
   A2A 에이전트 결과 충돌 해결:
   
   Data Cleaning Agent: "데이터에 {cleaning_issues}개 문제 발견"
   EDA Tools Agent: "데이터 품질이 {eda_quality_score}점"
   
   충돌 분석:
   1. 각 에이전트의 분석 방법과 기준이 다른가?
   2. 어떤 결과가 더 신뢰할 만한가?
   3. 사용자에게 어떻게 설명할 것인가?
   
   일관된 해석을 제시하세요.
   """
   ```

3. WHEN integrating agent outputs THEN the system SHALL create unified data artifacts that combine all agent contributions
4. WHEN presenting results THEN the system SHALL attribute insights to specific agents while maintaining narrative coherence
5. WHEN agent results are incomplete THEN the system SHALL identify gaps and suggest additional analysis

### Requirement 23: A2A Agent Performance Optimization and Caching

**User Story:** As a system user, I want fast response times even when multiple A2A agents are involved, so that complex analysis doesn't become prohibitively slow.

#### Acceptance Criteria

1. WHEN executing A2A workflows THEN the system SHALL implement intelligent caching of agent results
2. WHEN similar queries are processed THEN the system SHALL reuse cached A2A agent outputs when appropriate
3. WHEN agents can run in parallel THEN the system SHALL execute them concurrently to minimize total processing time
4. WHEN agents have dependencies THEN the system SHALL optimize execution order to minimize waiting time
5. WHEN system load is high THEN the system SHALL implement load balancing across available agent instances

### Requirement 24: A2A Agent Error Handling and Resilience

**User Story:** As a user, I want the system to handle A2A agent failures gracefully, so that one failing agent doesn't break the entire analysis.

#### Acceptance Criteria

1. WHEN A2A agents fail THEN the system SHALL continue analysis with available agents and inform user of limitations
2. WHEN agent timeouts occur THEN the system SHALL implement progressive timeout strategies (5s → 15s → 30s)
3. WHEN agent responses are malformed THEN the system SHALL attempt to parse partial results and request clarification
4. WHEN critical agents fail THEN the system SHALL suggest alternative approaches or manual steps
5. WHEN agents recover THEN the system SHALL automatically reintegrate them into future workflows

### Requirement 25: A2A Agent Capability Enhancement

**User Story:** As a developer, I want the Universal Engine to enhance A2A agent capabilities through intelligent prompting and context provision, so that agents perform better than they would in isolation.

#### Acceptance Criteria

1. WHEN calling A2A agents THEN the system SHALL provide enhanced context from meta-reasoning analysis:
   ```python
   enhanced_agent_request = {
       'query': original_query,
       'data': processed_data,
       'context': {
           'user_expertise_level': user_profile.expertise,
           'domain_context': meta_analysis.domain_context,
           'analysis_goals': meta_analysis.inferred_goals,
           'previous_agent_results': workflow_results,
           'quality_requirements': meta_analysis.quality_expectations
       }
   }
   ```

2. WHEN agents need clarification THEN the system SHALL use Universal Engine reasoning to provide intelligent responses
3. WHEN agents produce suboptimal results THEN the system SHALL provide feedback and request improvements
4. WHEN agents suggest follow-up actions THEN the system SHALL evaluate suggestions through meta-reasoning
5. WHEN agents learn from interactions THEN the system SHALL coordinate learning across the agent ecosystem

### Requirement 26: Security and Privacy Protection

**User Story:** As a user concerned about data privacy, I want assurance that the learning system protects my data while still improving the service, so that I can use the system with confidence.

#### Acceptance Criteria

1. WHEN processing user data THEN the system SHALL ensure all data remains secure and is not shared inappropriately
2. WHEN learning from interactions THEN the system SHALL extract patterns without storing personally identifiable information
3. WHEN user requests data deletion THEN the system SHALL remove all associated data while preserving general learning
4. WHEN providing analysis THEN the system SHALL not reference other users' specific data or queries
5. WHEN system updates occur THEN privacy protections SHALL be maintained throughout the learning process