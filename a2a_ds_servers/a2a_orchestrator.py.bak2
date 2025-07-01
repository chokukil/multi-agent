 class CherryAI_v8_UniversalIntelligentOrchestrator(AgentExecutor):
    """
    CherryAI v8 - Universal Intelligent Orchestrator
    v7 장점 + A2A 프로토콜 극대화 + v8 실시간 스트리밍 통합
    """
    
    def __init__(self):
        super().__init__()
        
        # v7 핵심 컴포넌트 유지
        self.openai_client = self._initialize_openai_client()
        self.execution_monitor = ExecutionMonitor(self.openai_client)
        self.replanning_engine = A2AEnhancedReplanningEngine(self.openai_client)
        
        # v8 A2A 프로토콜 극대화 컴포넌트
        self.a2a_discovery = A2AIntelligentAgentDiscovery()
        self.intelligent_matcher = IntelligentAgentMatcher(self.openai_client)
        self.discovered_agents = {}
        
        # v8 실시간 스트리밍 컴포넌트
        self.streaming_updater = None  # execute에서 초기화
        
        logger.info("🚀 CherryAI v8 Universal Intelligent Orchestrator 초기화 완료")
    
    def _initialize_openai_client(self) -> Optional[AsyncOpenAI]:
        """OpenAI 클라이언트 초기화"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음")
                return None
            
            return AsyncOpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
            return None
    
    async def execute(self, context: RequestContext) -> None:
        """v8 메인 실행 로직"""
        
        # v8 실시간 스트리밍 초기화
        self.streaming_updater = RealTimeStreamingTaskUpdater(
            context.task_updater.task_store,
            context.task_updater.task_id,
            context.task_updater.event_queue
        )
        
        try:
            # 사용자 입력 추출
            user_input = self._extract_user_input(context)
            if not user_input:
                await self.streaming_updater.stream_update("❌ 사용자 입력을 찾을 수 없습니다.")
                return
            
            logger.info(f"📝 사용자 요청: {user_input[:100]}...")
            
            # Phase 1: A2A 동적 에이전트 발견
            await self.streaming_updater.stream_update("🔍 A2A 프로토콜로 에이전트를 발견하고 있습니다...")
            self.discovered_agents = await self.a2a_discovery.discover_agents_dynamically()
            
            if not self.discovered_agents:
                await self.streaming_updater.stream_update("⚠️ 사용 가능한 에이전트를 찾을 수 없습니다.")
                return
            
            await self.streaming_updater.stream_update(
                f"✅ {len(self.discovered_agents)}개의 A2A 에이전트를 발견했습니다!"
            )
            
            # Phase 2: v7 적응적 복잡도 처리 (유지)
            await self.streaming_updater.stream_update("🧠 요청 복잡도를 분석하고 있습니다...")
            complexity_assessment = await self._assess_request_complexity(user_input)
            
            complexity_level = complexity_assessment.get('complexity_level', 'complex')
            await self.streaming_updater.stream_update(
                f"📊 요청 복잡도: {complexity_level} (점수: {complexity_assessment.get('complexity_score', 0)})"
            )
            
            # Phase 3: 복잡도별 처리 분기
            if complexity_level == 'simple':
                await self._handle_simple_request(user_input, complexity_assessment)
            elif complexity_level == 'single_agent':
                await self._handle_single_agent_request(user_input, complexity_assessment)
            else:
                await self._handle_complex_request(user_input, complexity_assessment)
                
        except Exception as e:
            logger.error(f"❌ v8 실행 중 오류: {e}")
            await self.streaming_updater.stream_update(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
    
    def _extract_user_input(self, context: RequestContext) -> str:
        """사용자 입력 추출"""
        try:
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part.root, 'kind') and part.root.kind == 'text':
                        return part.root.text
                    elif hasattr(part.root, 'type') and part.root.type == 'text':
                        return part.root.text
            return ""
        except Exception as e:
            logger.error(f"사용자 입력 추출 실패: {e}")
            return ""
    
    async def _assess_request_complexity(self, user_input: str) -> Dict:
        """v7 적응적 복잡도 처리 시스템 (유지)"""
        
        if not self.openai_client:
            return self._create_fallback_complexity_assessment(user_input)
        
        complexity_prompt = f"""
        다음 사용자 요청의 복잡도를 분석하세요:
        "{user_input}"
        
        복잡도 기준:
        1. **Simple (1-3점)**: 간단한 질문, 정보 요청, 기본 설명
        2. **Single Agent (4-6점)**: 특정 도구나 에이전트 하나로 해결 가능
        3. **Complex (7-10점)**: 여러 단계, 다중 에이전트, 종합적 분석 필요
        
        평가 요소:
        - 요청의 구체성과 범위
        - 필요한 데이터 처리 단계 수
        - 요구되는 전문성 수준
        - 예상 소요 시간과 리소스
        
        JSON 응답:
        {{
            "complexity_level": "simple|single_agent|complex",
            "complexity_score": 1-10,
            "reasoning": "복잡도 판단 근거",
            "estimated_steps": 예상_단계_수,
            "required_expertise": ["필요한_전문_영역들"],
            "processing_approach": "권장_처리_방식"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": complexity_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=30.0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"복잡도 분석 실패: {e}")
            return self._create_fallback_complexity_assessment(user_input)
    
    def _create_fallback_complexity_assessment(self, user_input: str) -> Dict:
        """폴백 복잡도 평가"""
        # 간단한 휴리스틱 기반 평가
        word_count = len(user_input.split())
        question_marks = user_input.count('?')
        keywords = ['분석', '예측', '모델링', '시각화', '비교', '추천']
        keyword_count = sum(1 for keyword in keywords if keyword in user_input)
        
        if word_count < 10 and question_marks > 0:
            return {
                'complexity_level': 'simple',
                'complexity_score': 2,
                'reasoning': '짧은 질문',
                'estimated_steps': 1,
                'required_expertise': ['general'],
                'processing_approach': 'direct_answer'
            }
        elif keyword_count == 1 and word_count < 30:
            return {
                'complexity_level': 'single_agent',
                'complexity_score': 5,
                'reasoning': '단일 작업 요청',
                'estimated_steps': 2,
                'required_expertise': ['data_analysis'],
                'processing_approach': 'single_agent'
            }
        else:
            return {
                'complexity_level': 'complex',
                'complexity_score': 8,
                'reasoning': '복합적 요청',
                'estimated_steps': 5,
                'required_expertise': ['data_analysis', 'visualization'],
                'processing_approach': 'multi_agent'
            }
    
    async def _handle_simple_request(self, user_input: str, complexity_assessment: Dict):
        """Simple 요청 처리 - 즉시 답변"""
        
        await self.streaming_updater.stream_update("💡 간단한 요청으로 판단되어 즉시 답변을 생성합니다...")
        
        if not self.openai_client:
            await self.streaming_updater.stream_response_progressively(
                "죄송합니다. OpenAI API가 설정되지 않아 답변을 생성할 수 없습니다."
            )
            return
        
        simple_prompt = f"""
        다음 질문에 명확하고 도움이 되는 답변을 제공하세요:
        "{user_input}"
        
        답변 요구사항:
        - 정확하고 구체적인 정보 제공
        - 필요시 예시나 설명 포함
        - 추가 도움이 필요한 경우 안내
        """
        
        try:
            # LLM 스트리밍 응답
            stream = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": simple_prompt}],
                stream=True,
                temperature=0.5,
                timeout=60.0
            )
            
            # 실시간 스트리밍으로 답변 전달
            final_response = await self.streaming_updater.stream_llm_response_realtime(stream)
            
            logger.info(f"✅ Simple 요청 처리 완료: {len(final_response)} 문자")
            
        except Exception as e:
            logger.error(f"Simple 요청 처리 실패: {e}")
            await self.streaming_updater.stream_response_progressively(
                f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            )
    
    async def _handle_single_agent_request(self, user_input: str, complexity_assessment: Dict):
        """Single Agent 요청 처리 - 최적 에이전트 직접 실행"""
        
        await self.streaming_updater.stream_update("🎯 단일 에이전트로 처리 가능한 요청입니다...")
        
        # 1. 사용자 의도 정밀 추출
        user_intent = await self._extract_user_intent_precisely(user_input)
        
        # 2. A2A 지능적 에이전트 매칭
        agent_matching = await self.intelligent_matcher.match_agents_to_intent(
            user_intent, self.discovered_agents
        )
        
        if not agent_matching.get('primary_workflow'):
            await self.streaming_updater.stream_response_progressively(
                "적합한 에이전트를 찾을 수 없습니다. 복합 처리로 전환합니다."
            )
            await self._handle_complex_request(user_input, complexity_assessment)
            return
        
        # 3. 최적 에이전트 실행
        best_agent = agent_matching['primary_workflow'][0]
        agent_name = best_agent['agent_name']
        
        await self.streaming_updater.stream_update(f"🤖 {agent_name} 에이전트를 실행합니다...")
        
        # 4. 정밀한 에이전트 지시 생성
        precise_instruction = await self._create_precise_agent_instruction(
            user_intent, best_agent, context={}
        )
        
        # 5. 에이전트 실행
        agent_result = await self._execute_single_agent(
            agent_name, precise_instruction, user_intent
        )
        
        # 6. 결과 검증 및 최종 응답
        if agent_result.get('status') == 'success':
            final_response = await self._create_evidence_based_response(
                [agent_result], user_intent, agent_matching
            )
            await self.streaming_updater.stream_with_sections(final_response)
        else:
            await self.streaming_updater.stream_response_progressively(
                f"에이전트 실행에 실패했습니다. 오류: {agent_result.get('error', '알 수 없는 오류')}"
            )
    
    async def _handle_complex_request(self, user_input: str, complexity_assessment: Dict):
        """Complex 요청 처리 - v7 전체 워크플로우 + A2A 강화"""
        
        await self.streaming_updater.stream_update("🔄 복합적인 요청으로 전체 워크플로우를 실행합니다...")
        
        # 1. v7 정밀한 사용자 의도 추출 (유지)
        await self.streaming_updater.stream_update("🎯 사용자 의도를 정밀하게 분석하고 있습니다...")
        user_intent = await self._extract_user_intent_precisely(user_input)
        
        # 2. A2A 지능적 에이전트 매칭 (신규)
        await self.streaming_updater.stream_update("🤖 최적의 에이전트 조합을 찾고 있습니다...")
        agent_matching = await self.intelligent_matcher.match_agents_to_intent(
            user_intent, self.discovered_agents
        )
        
        if not agent_matching.get('primary_workflow'):
            await self.streaming_updater.stream_response_progressively(
                "적합한 에이전트를 찾을 수 없습니다."
            )
            return
        
        # 3. 실행 계획 생성
        execution_plan = agent_matching['primary_workflow']
        await self.streaming_updater.stream_update(
            f"📋 {len(execution_plan)}단계 실행 계획을 수립했습니다."
        )
        
        # 4. 순차 실행 with 리플래닝
        execution_results = []
        execution_context = {'user_intent': user_intent}
        
        for step_idx, step in enumerate(execution_plan):
            agent_name = step['agent_name']
            
            await self.streaming_updater.stream_update(
                f"🔄 단계 {step_idx + 1}/{len(execution_plan)}: {agent_name} 실행 중..."
            )
            
            # 정밀한 에이전트 지시 생성
            precise_instruction = await self._create_precise_agent_instruction(
                user_intent, step, execution_context
            )
            
            # 에이전트 실행
            agent_result = await self._execute_single_agent(
                agent_name, precise_instruction, user_intent
            )
            
            # 결과 검증
            validation_result = await self._validate_agent_response(
                agent_result, step, user_intent
            )
            
            agent_result['validation'] = validation_result
            execution_results.append({
                'step': step_idx + 1,
                'agent': agent_name,
                'result': agent_result
            })
            
            # 리플래닝 필요성 검토
            should_replan = await self.execution_monitor.should_replan(
                step_idx, agent_result, execution_plan[step_idx+1:], user_intent
            )
            
            if should_replan.get('should_replan'):
                await self.streaming_updater.stream_update(
                    f"🔄 리플래닝이 필요합니다: {should_replan.get('reason')}"
                )
                
                # A2A 강화 리플래닝
                replan_result = await self.replanning_engine.create_a2a_enhanced_replan(
                    should_replan, 
                    {'failed_agent': agent_name if agent_result.get('status') == 'failed' else None},
                    execution_plan[step_idx+1:],
                    user_intent,
                    execution_results,
                    self.discovered_agents
                )
                
                if replan_result.get('strategy') != 'continue':
                    execution_plan = execution_plan[:step_idx+1] + replan_result.get('steps', [])
                    await self.streaming_updater.stream_update(
                        f"📋 실행 계획이 수정되었습니다. 전략: {replan_result.get('strategy')}"
                    )
            
            # 실행 컨텍스트 업데이트
            execution_context[f'step_{step_idx+1}_result'] = agent_result
        
        # 5. 최종 종합 응답 생성
        await self.streaming_updater.stream_update("📝 최종 종합 분석을 생성하고 있습니다...")
        
        final_response = await self._create_evidence_based_response(
            execution_results, user_intent, agent_matching
        )
        
        # 6. v8 실시간 스트리밍으로 최종 응답 전달
        await self.streaming_updater.stream_with_sections(final_response)
        
        logger.info(f"✅ Complex 요청 처리 완료: {len(execution_results)}단계 실행")
    
    async def _extract_user_intent_precisely(self, user_input: str) -> Dict:
        """v7 정밀한 사용자 의도 추출 (유지)"""
        
        if not self.openai_client:
            return self._create_fallback_intent(user_input)
        
        intent_prompt = f"""
        다음 사용자 요청을 정밀하게 분석하여 의도를 추출하세요:
        "{user_input}"
        
        추출할 정보:
        1. **주요 목표**: 사용자가 궁극적으로 원하는 것
        2. **행동 유형**: analyze|verify|recommend|diagnose|predict|compare|explain
        3. **분석 초점**: 어떤 측면에 집중해야 하는가
        4. **기대 결과**: 사용자가 기대하는 구체적 산출물
        5. **긴급성**: immediate|normal|thorough
        6. **전문성 수준**: beginner|intermediate|expert
        
        JSON 응답:
        {{
            "main_goal": "구체적인 목표",
            "action_type": "행동_유형",
            "analysis_focus": ["초점_영역들"],
            "expected_outcomes": ["기대_결과들"],
            "urgency": "긴급성_수준",
            "expertise_level": "전문성_수준",
            "domain": "주요_도메인",
            "specific_requirements": ["구체적_요구사항들"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": intent_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=45.0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"의도 추출 실패: {e}")
            return self._create_fallback_intent(user_input)
    
    def _create_fallback_intent(self, user_input: str) -> Dict:
        """폴백 의도 추출"""
        return {
            "main_goal": user_input,
            "action_type": "analyze",
            "analysis_focus": ["general"],
            "expected_outcomes": ["analysis_result"],
            "urgency": "normal",
            "expertise_level": "intermediate",
            "domain": "data_analysis",
            "specific_requirements": []
        }
    
    async def _create_precise_agent_instruction(self, 
                                              user_intent: Dict,
                                              agent_step: Dict,
                                              context: Dict) -> str:
        """v7 정밀한 에이전트 지시 생성 (A2A 강화)"""
        
        agent_name = agent_step.get('agent_name', '')
        skill_id = agent_step.get('skill_id', '')
        
        # A2A Agent Card에서 실제 스킬 정보 가져오기
        agent_info = self.discovered_agents.get(agent_name, {})
        skill_info = {}
        if hasattr(agent_info, 'skills') and skill_id in agent_info.skills:
            skill_info = agent_info.skills[skill_id]
        
        if not self.openai_client:
            return self._create_fallback_instruction(user_intent, agent_name)
        
        instruction_prompt = f"""
        A2A 에이전트를 위한 정밀한 작업 지시를 생성하세요.
        
        에이전트 정보:
        - 이름: {agent_name}
        - 사용할 스킬: {skill_id}
        - 스킬 설명: {skill_info.get('description', '')}
        - 스킬 예시: {skill_info.get('examples', [])}
        
        사용자 의도:
        {json.dumps(user_intent, ensure_ascii=False, indent=2)}
        
        실행 컨텍스트:
        {json.dumps({k: str(v)[:200] for k, v in context.items()}, ensure_ascii=False, indent=2)}
        
        다음 요구사항을 충족하는 구체적 지시를 생성하세요:
        1. **명확한 작업 정의**: 정확히 무엇을 해야 하는지
        2. **입력 데이터 명시**: 어떤 데이터를 사용할지
        3. **출력 형식 지정**: 결과를 어떤 형태로 제공할지
        4. **품질 기준**: 결과가 만족해야 할 조건
        5. **에러 처리**: 문제 발생시 대응 방법
        
        에이전트가 이해하기 쉬운 한국어로 작성하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": instruction_prompt}],
                temperature=0.4,
                timeout=45.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"정밀 지시 생성 실패: {e}")
            return self._create_fallback_instruction(user_intent, agent_name)
    
    def _create_fallback_instruction(self, user_intent: Dict, agent_name: str) -> str:
        """폴백 지시 생성"""
        main_goal = user_intent.get('main_goal', '')
        action_type = user_intent.get('action_type', 'analyze')
        
        return f"""
        {agent_name} 에이전트 작업 요청:
        
        목표: {main_goal}
        작업 유형: {action_type}
        
        요청사항:
        1. 제공된 데이터를 {action_type}하세요
        2. 결과를 명확하고 구체적으로 제공하세요
        3. 문제가 있으면 상세한 오류 정보를 포함하세요
        """
    
    async def _execute_single_agent(self, agent_name: str, instruction: str, user_intent: Dict) -> Dict:
        """단일 에이전트 실행"""
        
        # A2A 에이전트 정보 가져오기
        agent_info = self.discovered_agents.get(agent_name)
        if not agent_info:
            return {
                'status': 'failed',
                'error': f'에이전트 {agent_name}을 찾을 수 없습니다',
                'result': None
            }
        
        try:
            # A2A 클라이언트로 에이전트 호출
            async with httpx.AsyncClient(timeout=120.0) as client:
                a2a_client = A2AClient(httpx_client=client, base_url=agent_info.url)
                
                # A2A 표준 메시지 생성
                message = {
                    "parts": [{"kind": "text", "text": instruction}],
                    "messageId": f"msg_{int(time.time())}",
                    "role": "user"
                }
                
                # 에이전트 실행
                response = await a2a_client.send_message(message)
                
                if response and hasattr(response, 'parts') and response.parts:
                    result_text = ""
                    for part in response.parts:
                        if hasattr(part.root, 'text'):
                            result_text += part.root.text
                    
                    return {
                        'status': 'success',
                        'result': result_text,
                        'agent_name': agent_name,
                        'instruction': instruction
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': '에이전트에서 응답을 받지 못했습니다',
                        'result': None
                    }
                    
        except Exception as e:
            logger.error(f"에이전트 {agent_name} 실행 실패: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'result': None
            }
    
    async def _validate_agent_response(self, agent_result: Dict, step: Dict, user_intent: Dict) -> Dict:
        """에이전트 응답 검증"""
        
        if agent_result.get('status') != 'success':
            return {
                'is_valid': False,
                'confidence': 0.0,
                'warnings': ['에이전트 실행 실패'],
                'evidence_score': 0.0
            }
        
        result_text = agent_result.get('result', '')
        
        # 기본적인 검증
        if not result_text or len(result_text.strip()) < 10:
            return {
                'is_valid': False,
                'confidence': 0.2,
                'warnings': ['응답이 너무 짧거나 비어있음'],
                'evidence_score': 0.1
            }
        
        # 오류 메시지 검출
        error_indicators = ['오류', 'error', '실패', '문제', '불가능']
        if any(indicator in result_text.lower() for indicator in error_indicators):
            return {
                'is_valid': False,
                'confidence': 0.3,
                'warnings': ['오류 메시지 포함'],
                'evidence_score': 0.2
            }
        
        return {
            'is_valid': True,
            'confidence': 0.8,
            'warnings': [],
            'evidence_score': 0.8
        }
    
    async def _create_evidence_based_response(self, 
                                            execution_results: List[Dict],
                                            user_intent: Dict,
                                            agent_matching: Dict) -> str:
        """증거 기반 최종 응답 생성"""
        
        if not self.openai_client:
            return self._create_fallback_final_response(execution_results, user_intent)
        
        # 실행 결과 요약
        results_summary = []
        for exec_result in execution_results:
            result_info = {
                'step': exec_result['step'],
                'agent': exec_result['agent'],
                'status': exec_result['result'].get('status'),
                'content': exec_result['result'].get('result', '')[:500],  # 처음 500자만
                'validation': exec_result['result'].get('validation', {})
            }
            results_summary.append(result_info)
        
        synthesis_prompt = f"""
        다음 A2A 에이전트 실행 결과들을 종합하여 사용자 요청에 대한 완전한 답변을 생성하세요.
        
        사용자 의도:
        {json.dumps(user_intent, ensure_ascii=False, indent=2)}
        
        에이전트 실행 결과:
        {json.dumps(results_summary, ensure_ascii=False, indent=2)}
        
        응답 요구사항:
        1. **사용자 목표 달성**: {user_intent.get('main_goal', '')}에 직접적으로 답변
        2. **증거 기반**: 실제 에이전트 결과만 사용, 추측 금지
        3. **구조화된 답변**: 명확한 섹션과 헤더 사용
        4. **실행 가능한 인사이트**: 구체적이고 실용적인 정보 제공
        5. **한계점 명시**: 부족한 부분이나 추가 분석 필요 사항 표시
        
        Markdown 형식으로 작성하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.5,
                timeout=90.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"최종 응답 생성 실패: {e}")
            return self._create_fallback_final_response(execution_results, user_intent)
    
    def _create_fallback_final_response(self, execution_results: List[Dict], user_intent: Dict) -> str:
        """폴백 최종 응답"""
        response = f"# {user_intent.get('main_goal', '분석 결과')}\n\n"
        
        for exec_result in execution_results:
            agent_name = exec_result['agent']
            result = exec_result['result']
            
            response += f"## {agent_name} 결과\n\n"
            
            if result.get('status') == 'success':
                response += f"{result.get('result', '결과 없음')}\n\n"
            else:
                response += f"⚠️ 실행 실패: {result.get('error', '알 수 없는 오류')}\n\n"
        
        response += "---\n\n*CherryAI v8 Universal Intelligent Orchestrator로 생성된 분석 결과입니다.*"
        
        return response
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info("🛑 CherryAI v8 작업이 취소되었습니다")
        if self.streaming_updater:
            await self.streaming_updater.stream_update("🛑 작업이 사용자에 의해 취소되었습니다.")


# Agent Card 정의
def create_agent_card() -> AgentCard:
    """CherryAI v8 Agent Card 생성"""
    return AgentCard(
        name="CherryAI v8 Universal Intelligent Orchestrator",
        description="v7 장점 + A2A 프로토콜 극대화 + v8 실시간 스트리밍을 통합한 범용 지능형 오케스트레이터",
        skills=[
            AgentSkill(
                id="universal_analysis",
                name="범용 데이터 분석",
                description="모든 종류의 데이터 분석 요청을 A2A 에이전트들과 협력하여 처리",
                tags=["data-analysis", "orchestration", "a2a", "streaming"],
                examples=[
                    "데이터셋의 패턴과 이상치를 분석해주세요",
                    "매출 데이터의 트렌드를 예측해주세요", 
                    "고객 세분화 분석을 수행해주세요"
                ]
            ),
            AgentSkill(
                id="intelligent_replanning",
                name="지능적 리플래닝",
                description="실행 중 문제 발생시 A2A 에이전트 정보를 활용한 동적 계획 수정",
                tags=["replanning", "adaptive", "a2a-discovery"],
                examples=[
                    "에이전트 실패시 대체 에이전트로 자동 전환",
                    "새로운 발견에 따른 분석 방향 조정"
                ]
            ),
            AgentSkill(
                id="realtime_streaming",
                name="실시간 스트리밍",
                description="분석 과정과 결과를 실시간으로 스트리밍하여 사용자 경험 향상",
                tags=["streaming", "realtime", "progressive"],
                examples=[
                    "분석 진행 상황을 실시간으로 확인",
                    "결과를 점진적으로 받아보기"
                ]
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            supportsAuthenticatedExtendedCard=False
        )
    )


# 서버 실행 부분
if __name__ == "__main__":
    # A2A 서버 설정
    task_store = InMemoryTaskStore()
    event_queue = EventQueue()
    
    # 오케스트레이터 생성
    orchestrator = CherryAI_v8_UniversalIntelligentOrchestrator()
    
    # 요청 핸들러 설정
    request_handler = DefaultRequestHandler(
        agent_executor=orchestrator,
        task_store=task_store,
        event_queue=event_queue
    )
    
    # A2A 애플리케이션 생성
    app = A2AStarletteApplication(
        agent_card=create_agent_card(),
        request_handler=request_handler
    )
    
    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    ) 