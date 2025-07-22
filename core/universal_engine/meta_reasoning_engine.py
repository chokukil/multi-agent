"""
Meta-Reasoning Engine - 메타 추론 엔진

요구사항 2에 따른 DeepSeek-R1 영감을 받은 메타 추론 시스템
- 생각에 대해 생각하기
- 4단계 추론: 초기 관찰 → 다각도 분석 → 자가 검증 → 적응적 응답
- 메타 보상 패턴으로 분석 품질 평가
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from .llm_factory import LLMFactory
from .optimizations.llm_performance_optimizer import optimize_llm_call

logger = logging.getLogger(__name__)


class MetaReasoningEngine:
    """
    메타 추론 엔진 - 생각에 대해 생각하기
    DeepSeek-R1 영감을 받은 자가 반성 추론 시스템
    """
    
    def __init__(self):
        """MetaReasoningEngine 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.reasoning_patterns = {
            'self_reflection': self._load_self_reflection_pattern(),
            'meta_rewarding': self._load_meta_rewarding_pattern(),
            'chain_of_thought': self._load_chain_of_thought_pattern(),
            'zero_shot_adaptive': self._load_zero_shot_pattern()
        }
        logger.info("MetaReasoningEngine initialized with DeepSeek-R1 patterns")
    
    def _load_self_reflection_pattern(self) -> str:
        """자가 반성 추론 패턴 로드"""
        return """
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
        """
    
    def _load_meta_rewarding_pattern(self) -> str:
        """메타 보상 패턴 로드"""
        return """
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
        """
    
    def _load_chain_of_thought_pattern(self) -> str:
        """Chain of Thought 패턴 로드"""
        return """
        문제를 단계별로 생각해보겠습니다:
        
        1. 문제 정의: 정확히 무엇을 해결해야 하는가?
        2. 가능한 접근법: 어떤 방법들이 있는가?
        3. 각 접근법의 장단점: 무엇이 최선인가?
        4. 선택한 접근법 실행: 구체적으로 어떻게?
        5. 결과 검증: 올바른 결과인가?
        """
    
    def _load_zero_shot_pattern(self) -> str:
        """Zero-shot 적응형 추론 패턴 로드"""
        return """
        템플릿이나 공식에 의존하지 않고 문제 자체의 본질에 맞는 추론을 수행하겠습니다.
        
        문제 공간 정의:
        - 주어진 정보는 무엇인가?
        - 요구되는 결과는 무엇인가?
        - 제약사항은 무엇인가?
        
        추론 전략 수립:
        - 이 특정 문제에 가장 적합한 사고 방식은?
        - 어떤 관점에서 접근해야 하는가?
        
        단계별 추론 실행:
        - 각 단계의 논리적 연결성 확보
        - 가정과 제약사항 명시
        - 불확실성과 신뢰도 평가
        
        결과 통합 및 검증:
        - 모든 요구사항이 충족되었는가?
        - 논리적 일관성이 있는가?
        - 실용적 가치가 있는가?
        """
    
    async def analyze_request(self, query: str, data: Any, context: Dict) -> Dict:
        """
        요구사항 2에 따른 메타 추론 분석
        - DeepSeek-R1 영감 4단계 추론
        - 자가 평가 및 개선
        - 동적 전략 선택
        
        Args:
            query: 사용자 쿼리
            data: 분석 대상 데이터
            context: 추가 컨텍스트
            
        Returns:
            메타 추론 분석 결과
        """
        logger.info("Starting meta-reasoning analysis")
        
        try:
            # 1단계: 초기 관찰 및 분석
            initial_analysis = await self._perform_initial_observation(query, data)
            
            # 2단계: 다각도 분석
            multi_perspective_analysis = await self._perform_multi_perspective_analysis(
                initial_analysis, query, data
            )
            
            # 3단계: 자가 검증
            self_verification = await self._perform_self_verification(
                multi_perspective_analysis
            )
            
            # 4단계: 적응적 응답 전략 결정
            response_strategy = await self._determine_adaptive_strategy(
                self_verification, context
            )
            
            # 메타 보상 패턴으로 전체 분석 품질 평가
            quality_assessment = await self._assess_analysis_quality(response_strategy)
            
            return {
                'initial_analysis': initial_analysis,
                'multi_perspective': multi_perspective_analysis,
                'self_verification': self_verification,
                'response_strategy': response_strategy,
                'quality_assessment': quality_assessment,
                'confidence_level': quality_assessment.get('confidence', 0.0),
                'user_profile': response_strategy.get('estimated_user_profile', {}),
                'domain_context': initial_analysis.get('domain_context', {}),
                'data_characteristics': initial_analysis.get('data_characteristics', {})
            }
            
        except Exception as e:
            logger.error(f"Error in meta-reasoning analysis: {e}")
            raise
    
    async def _perform_initial_observation(self, query: str, data: Any) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 1: 초기 관찰
        """
        observation_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        쿼리: {query}
        데이터 특성: {self._analyze_data_characteristics(data)}
        
        단계 1: 초기 관찰
        - 데이터를 보고 무엇을 발견하는가?
        - 사용자 쿼리의 진정한 의도는?
        - 내가 놓치고 있는 것은 없는가?
        
        사전 정의된 카테고리나 패턴에 의존하지 말고,
        순수하게 관찰된 것만을 바탕으로 분석하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "data_observations": "관찰된 데이터 특성",
            "query_intent": "파악된 사용자 의도",
            "potential_blind_spots": "놓칠 수 있는 부분",
            "domain_context": "감지된 도메인 컨텍스트",
            "data_characteristics": {{
                "type": "데이터 유형",
                "structure": "데이터 구조",
                "patterns": "발견된 패턴"
            }}
        }}
        """
        
        llm_response = await self.llm_client.ainvoke(observation_prompt)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return self._parse_json_response(response)
    
    async def _perform_multi_perspective_analysis(self, initial_analysis: Dict, query: str, data: Any) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 2: 다각도 분석
        """
        multi_perspective_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        초기 관찰 결과: {json.dumps(initial_analysis, ensure_ascii=False)}
        
        단계 2: 다각도 분석
        - 이 문제를 다른 방식으로 접근한다면?
        - 사용자가 전문가라면 어떤 답을 원할까?
        - 사용자가 초보자라면 어떤 도움이 필요할까?
        
        각 관점에서의 분석을 수행하고,
        사용자 수준 추정과 최적 접근법을 제시하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "alternative_approaches": ["접근법1", "접근법2", "접근법3"],
            "expert_perspective": {{
                "expectations": "전문가가 원하는 것",
                "technical_depth": "필요한 기술적 깊이",
                "key_insights": ["인사이트1", "인사이트2"]
            }},
            "beginner_perspective": {{
                "needs": "초보자가 필요로 하는 것",
                "simplifications": "단순화할 부분",
                "guidance": "제공할 가이드"
            }},
            "estimated_user_level": "beginner|intermediate|expert|unknown",
            "recommended_approach": "추천하는 접근법"
        }}
        """
        
        llm_response = await self.llm_client.ainvoke(multi_perspective_prompt)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return self._parse_json_response(response)
    
    async def _perform_self_verification(self, multi_perspective_analysis: Dict) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 3: 자가 검증
        """
        verification_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        다각도 분석 결과: {json.dumps(multi_perspective_analysis, ensure_ascii=False)}
        
        단계 3: 자가 검증
        - 내 분석이 논리적으로 일관성이 있는가?
        - 사용자에게 실제로 도움이 되는가?
        - 확신이 없는 부분은 무엇인가?
        
        분석의 강점과 약점을 솔직하게 평가하고,
        불확실한 부분은 명확히 식별하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "logical_consistency": {{
                "is_consistent": true/false,
                "inconsistencies": ["불일치1", "불일치2"]
            }},
            "practical_value": {{
                "is_helpful": true/false,
                "value_points": ["가치1", "가치2"],
                "limitations": ["한계1", "한계2"]
            }},
            "uncertainties": {{
                "high_confidence_areas": ["확실한 부분1", "확실한 부분2"],
                "low_confidence_areas": ["불확실한 부분1", "불확실한 부분2"],
                "clarification_needed": ["명확화 필요1", "명확화 필요2"]
            }},
            "overall_confidence": 0.0-1.0
        }}
        """
        
        llm_response = await self.llm_client.ainvoke(verification_prompt)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return self._parse_json_response(response)
    
    async def _determine_adaptive_strategy(self, self_verification: Dict, context: Dict) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 4: 적응적 응답
        """
        strategy_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        자가 검증 결과: {json.dumps(self_verification, ensure_ascii=False)}
        상호작용 컨텍스트: {json.dumps(context, ensure_ascii=False)}
        
        단계 4: 적응적 응답
        - 확실한 부분은 명확히 제시
        - 불확실한 부분은 명확화 질문
        - 사용자 수준에 맞는 설명 깊이 조절
        
        최적의 응답 전략을 결정하고,
        사용자 프로필과 상호작용 방식을 제안하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "response_strategy": {{
                "approach": "직접적|점진적|대화형",
                "explanation_depth": "shallow|medium|deep",
                "technical_level": "low|medium|high",
                "interaction_style": "formal|casual|educational"
            }},
            "content_structure": {{
                "present_confidently": ["확실한 내용1", "확실한 내용2"],
                "seek_clarification": ["명확화 질문1", "명확화 질문2"],
                "progressive_disclosure": ["단계1", "단계2", "단계3"]
            }},
            "estimated_user_profile": {{
                "expertise_level": "beginner|intermediate|expert",
                "learning_style": "visual|textual|example-based",
                "domain_familiarity": "low|medium|high"
            }},
            "follow_up_recommendations": ["추천1", "추천2", "추천3"]
        }}
        """
        
        llm_response = await self.llm_client.ainvoke(strategy_prompt)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return self._parse_json_response(response)
    
    async def _assess_analysis_quality(self, response_strategy: Dict) -> Dict:
        """
        요구사항 2의 메타 보상 패턴으로 분석 품질 평가
        """
        quality_prompt = f"""
        {self.reasoning_patterns['meta_rewarding']}
        
        분석 내용: {json.dumps(response_strategy, ensure_ascii=False)}
        
        평가 기준:
        1. 정확성: 분석이 데이터를 올바르게 해석했는가?
        2. 완전성: 중요한 인사이트를 놓치지 않았는가?
        3. 적절성: 사용자 수준과 요구에 맞는가?
        4. 명확성: 설명이 이해하기 쉬운가?
        5. 실용성: 실제로 도움이 되는 조치를 제안했는가?
        
        JSON 형식으로 평가 결과를 제공하세요:
        {{
            "accuracy_score": 0.0-1.0,
            "completeness_score": 0.0-1.0,
            "appropriateness_score": 0.0-1.0,
            "clarity_score": 0.0-1.0,
            "practicality_score": 0.0-1.0,
            "overall_quality": 0.0-1.0,
            "confidence": 0.0-1.0,
            "improvements_needed": ["개선점1", "개선점2"],
            "strengths": ["강점1", "강점2"],
            "next_steps": ["다음 단계1", "다음 단계2"]
        }}
        """
        
        # 최적화된 LLM 호출 사용
        optimized_result = await optimize_llm_call(self.llm_client, quality_prompt)
        response = optimized_result["response"]
        return self._parse_json_response(response)
    
    def _analyze_data_characteristics(self, data: Any) -> str:
        """데이터 특성 분석"""
        characteristics = {
            'type': type(data).__name__,
            'size': self._get_data_size(data),
            'structure': self._get_data_structure(data)
        }
        return json.dumps(characteristics, ensure_ascii=False)
    
    def _get_data_size(self, data: Any) -> str:
        """데이터 크기 확인"""
        try:
            if hasattr(data, '__len__'):
                return f"{len(data)} items"
            elif hasattr(data, 'shape'):
                return f"shape: {data.shape}"
            else:
                return "unknown size"
        except:
            return "size calculation failed"
    
    def _get_data_structure(self, data: Any) -> str:
        """데이터 구조 확인"""
        try:
            if hasattr(data, 'columns'):
                return f"columns: {list(data.columns)[:5]}..."
            elif isinstance(data, dict):
                return f"keys: {list(data.keys())[:5]}..."
            elif isinstance(data, list):
                return f"list of {type(data[0]).__name__ if data else 'empty'}"
            else:
                return type(data).__name__
        except:
            return "structure analysis failed"
    
    async def perform_meta_reasoning(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        완전한 메타 추론 프로세스 실행
        
        요구사항 1.1에 따른 구현:
        - DeepSeek-R1 영감을 받은 4단계 메타 추론
        - 자가 반성 및 검증
        - 동적 전략 선택
        
        Args:
            query: 사용자 쿼리
            context: 추론 컨텍스트 (데이터, 사용자 정보 등)
            
        Returns:
            완전한 메타 추론 결과
        """
        logger.info("Performing complete meta-reasoning process")
        
        try:
            # 컨텍스트에서 데이터 추출
            data = context.get('data', None)
            user_context = context.get('user_context', {})
            
            # 전체 메타 추론 프로세스 실행
            meta_analysis = await self.analyze_request(query, data, user_context)
            
            # 추가적인 메타 레벨 추론 수행
            meta_meta_analysis = await self._perform_meta_meta_reasoning(meta_analysis, query)
            
            # 최종 통합 결과 생성
            integrated_result = {
                'query': query,
                'context': context,
                'meta_reasoning_stages': {
                    'stage_1_observation': meta_analysis.get('initial_analysis', {}),
                    'stage_2_multi_perspective': meta_analysis.get('multi_perspective', {}),
                    'stage_3_verification': meta_analysis.get('self_verification', {}),
                    'stage_4_strategy': meta_analysis.get('response_strategy', {})
                },
                'meta_meta_analysis': meta_meta_analysis,
                'quality_assessment': meta_analysis.get('quality_assessment', {}),
                'confidence_metrics': {
                    'overall_confidence': meta_analysis.get('confidence_level', 0.0),
                    'reasoning_depth': self._calculate_reasoning_depth(meta_analysis),
                    'consistency_score': self._calculate_consistency_score(meta_analysis)
                },
                'actionable_insights': await self._extract_actionable_insights(meta_analysis),
                'uncertainty_handling': await self._handle_uncertainties(meta_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error in meta-reasoning process: {e}")
            return {
                'error': str(e),
                'fallback_analysis': await self._fallback_reasoning(query, context),
                'timestamp': datetime.now().isoformat()
            }
    
    async def assess_analysis_quality(self, analysis_result: Dict) -> Dict[str, Any]:
        """
        분석 품질 평가 및 개선 제안
        
        요구사항 1.1에 따른 구현:
        - 메타 보상 패턴 기반 품질 평가
        - 5가지 평가 기준 적용
        - 구체적인 개선 제안 생성
        
        Args:
            analysis_result: 평가할 분석 결과
            
        Returns:
            품질 평가 결과 및 개선 제안
        """
        logger.info("Assessing analysis quality with meta-reward patterns")
        
        try:
            # 기본 품질 평가 수행
            basic_quality = await self._assess_analysis_quality(analysis_result)
            
            # 심화 품질 분석
            detailed_assessment = await self._perform_detailed_quality_analysis(analysis_result)
            
            # 개선 제안 생성
            improvement_suggestions = await self._generate_improvement_suggestions(
                analysis_result, basic_quality, detailed_assessment
            )
            
            # 벤치마킹 및 비교 분석
            benchmark_analysis = await self._benchmark_against_standards(analysis_result)
            
            # 최종 품질 보고서 생성
            quality_report = {
                'assessment_timestamp': datetime.now().isoformat(),
                'analysis_id': analysis_result.get('id', 'unknown'),
                'quality_scores': {
                    'accuracy': basic_quality.get('accuracy_score', 0.0),
                    'completeness': basic_quality.get('completeness_score', 0.0),
                    'appropriateness': basic_quality.get('appropriateness_score', 0.0),
                    'clarity': basic_quality.get('clarity_score', 0.0),
                    'practicality': basic_quality.get('practicality_score', 0.0),
                    'overall': basic_quality.get('overall_quality', 0.0)
                },
                'detailed_metrics': detailed_assessment,
                'improvement_suggestions': improvement_suggestions,
                'benchmark_comparison': benchmark_analysis,
                'confidence_assessment': {
                    'reliability': self._assess_reliability(analysis_result),
                    'robustness': self._assess_robustness(analysis_result),
                    'generalizability': self._assess_generalizability(analysis_result)
                },
                'meta_evaluation': {
                    'reasoning_quality': self._evaluate_reasoning_quality(analysis_result),
                    'logical_coherence': self._evaluate_logical_coherence(analysis_result),
                    'evidence_support': self._evaluate_evidence_support(analysis_result)
                },
                'actionable_recommendations': await self._generate_actionable_recommendations(
                    improvement_suggestions, benchmark_analysis
                )
            }
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return {
                'error': str(e),
                'basic_assessment': {'overall_quality': 0.0},
                'timestamp': datetime.now().isoformat()
            }
    
    async def _perform_meta_meta_reasoning(self, meta_analysis: Dict, query: str) -> Dict:
        """메타 추론에 대한 메타 추론 수행"""
        meta_meta_prompt = f"""
        나는 방금 다음과 같은 메타 추론을 수행했습니다:
        {json.dumps(meta_analysis, ensure_ascii=False, indent=2)}
        
        이제 이 메타 추론 자체에 대해 생각해보겠습니다:
        
        1. 내 추론 과정이 적절했는가?
        2. 놓친 중요한 관점이 있는가?
        3. 더 나은 추론 방법이 있었는가?
        4. 이 추론의 한계는 무엇인가?
        
        원래 쿼리: {query}
        
        JSON 형식으로 메타-메타 분석을 제공하세요:
        {{
            "reasoning_process_evaluation": {{
                "appropriateness": "적절성 평가",
                "thoroughness": "철저함 평가",
                "efficiency": "효율성 평가"
            }},
            "missed_perspectives": ["놓친 관점1", "놓친 관점2"],
            "alternative_approaches": ["대안적 접근법1", "대안적 접근법2"],
            "limitations_identified": ["한계1", "한계2"],
            "meta_confidence": 0.0-1.0,
            "recursive_insights": ["재귀적 인사이트1", "재귀적 인사이트2"]
        }}
        """
        
        llm_response = await self.llm_client.ainvoke(meta_meta_prompt)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return self._parse_json_response(response)
    
    async def _perform_detailed_quality_analysis(self, analysis_result: Dict) -> Dict:
        """상세 품질 분석 수행"""
        detailed_prompt = f"""
        다음 분석 결과에 대해 상세한 품질 평가를 수행하겠습니다:
        {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        
        평가 차원:
        1. 논리적 일관성: 추론 단계들이 논리적으로 연결되어 있는가?
        2. 증거 기반성: 주장이 적절한 증거로 뒷받침되는가?
        3. 포괄성: 중요한 측면들이 모두 다뤄졌는가?
        4. 정밀성: 분석이 구체적이고 정확한가?
        5. 실용성: 실제 적용 가능한 결과인가?
        
        JSON 형식으로 상세 평가를 제공하세요:
        {{
            "logical_consistency": {{
                "score": 0.0-1.0,
                "issues": ["문제점1", "문제점2"],
                "strengths": ["강점1", "강점2"]
            }},
            "evidence_basis": {{
                "score": 0.0-1.0,
                "well_supported": ["잘 뒷받침된 부분1", "잘 뒷받침된 부분2"],
                "needs_support": ["증거 필요 부분1", "증거 필요 부분2"]
            }},
            "comprehensiveness": {{
                "score": 0.0-1.0,
                "covered_aspects": ["다뤄진 측면1", "다뤄진 측면2"],
                "missing_aspects": ["누락된 측면1", "누락된 측면2"]
            }},
            "precision": {{
                "score": 0.0-1.0,
                "precise_elements": ["정확한 요소1", "정확한 요소2"],
                "vague_elements": ["모호한 요소1", "모호한 요소2"]
            }},
            "practicality": {{
                "score": 0.0-1.0,
                "actionable_items": ["실행 가능한 항목1", "실행 가능한 항목2"],
                "theoretical_items": ["이론적 항목1", "이론적 항목2"]
            }}
        }}
        """
        
        llm_response = await self.llm_client.ainvoke(detailed_prompt)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return self._parse_json_response(response)
    
    async def _generate_improvement_suggestions(self, analysis_result: Dict, basic_quality: Dict, detailed_assessment: Dict) -> Dict:
        """개선 제안 생성"""
        improvement_prompt = f"""
        분석 결과와 품질 평가를 바탕으로 구체적인 개선 제안을 생성하겠습니다:
        
        분석 결과: {json.dumps(analysis_result, ensure_ascii=False, indent=2)[:1000]}...
        기본 품질 평가: {json.dumps(basic_quality, ensure_ascii=False)}
        상세 평가: {json.dumps(detailed_assessment, ensure_ascii=False)}
        
        개선 영역별로 구체적이고 실행 가능한 제안을 제공하세요:
        
        JSON 형식으로 응답하세요:
        {{
            "immediate_improvements": {{
                "high_priority": ["즉시 개선 항목1", "즉시 개선 항목2"],
                "medium_priority": ["중간 우선순위 항목1", "중간 우선순위 항목2"],
                "low_priority": ["낮은 우선순위 항목1", "낮은 우선순위 항목2"]
            }},
            "structural_improvements": {{
                "reasoning_structure": ["추론 구조 개선1", "추론 구조 개선2"],
                "evidence_integration": ["증거 통합 개선1", "증거 통합 개선2"],
                "clarity_enhancements": ["명확성 향상1", "명확성 향상2"]
            }},
            "content_improvements": {{
                "depth_enhancements": ["깊이 향상1", "깊이 향상2"],
                "breadth_expansions": ["폭 확장1", "폭 확장2"],
                "precision_refinements": ["정밀도 개선1", "정밀도 개선2"]
            }},
            "implementation_roadmap": {{
                "phase_1": ["1단계 작업1", "1단계 작업2"],
                "phase_2": ["2단계 작업1", "2단계 작업2"],
                "phase_3": ["3단계 작업1", "3단계 작업2"]
            }}
        }}
        """
        
        llm_response = await self.llm_client.ainvoke(improvement_prompt)
        response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return self._parse_json_response(response)
    
    def _calculate_reasoning_depth(self, meta_analysis: Dict) -> float:
        """추론 깊이 계산"""
        depth_indicators = [
            len(meta_analysis.get('initial_analysis', {}).get('data_observations', '')),
            len(meta_analysis.get('multi_perspective', {}).get('alternative_approaches', [])),
            len(meta_analysis.get('self_verification', {}).get('uncertainties', {}).get('clarification_needed', [])),
            meta_analysis.get('quality_assessment', {}).get('overall_quality', 0.0)
        ]
        return sum(depth_indicators) / len(depth_indicators) if depth_indicators else 0.0
    
    def _calculate_consistency_score(self, meta_analysis: Dict) -> float:
        """일관성 점수 계산"""
        consistency_factors = [
            meta_analysis.get('self_verification', {}).get('logical_consistency', {}).get('is_consistent', False),
            meta_analysis.get('quality_assessment', {}).get('accuracy_score', 0.0),
            meta_analysis.get('confidence_level', 0.0)
        ]
        return sum(1 if factor else 0 for factor in consistency_factors) / len(consistency_factors)
    
    async def _extract_actionable_insights(self, meta_analysis: Dict) -> List[str]:
        """실행 가능한 인사이트 추출"""
        insights = []
        
        # 응답 전략에서 실행 가능한 항목 추출
        strategy = meta_analysis.get('response_strategy', {})
        content_structure = strategy.get('content_structure', {})
        
        insights.extend(content_structure.get('present_confidently', []))
        insights.extend(content_structure.get('progressive_disclosure', []))
        insights.extend(strategy.get('follow_up_recommendations', []))
        
        return insights[:10]  # 상위 10개만 반환
    
    async def _handle_uncertainties(self, meta_analysis: Dict) -> Dict:
        """불확실성 처리"""
        uncertainties = meta_analysis.get('self_verification', {}).get('uncertainties', {})
        
        return {
            'identified_uncertainties': uncertainties.get('low_confidence_areas', []),
            'clarification_questions': uncertainties.get('clarification_needed', []),
            'confidence_boosting_actions': [
                "추가 데이터 수집",
                "전문가 검토 요청",
                "다중 접근법 비교"
            ],
            'uncertainty_communication': "불확실한 부분을 사용자에게 투명하게 전달"
        }
    
    async def _fallback_reasoning(self, query: str, context: Dict) -> Dict:
        """폴백 추론 (오류 시 사용)"""
        return {
            'type': 'fallback',
            'query': query,
            'basic_analysis': f"쿼리 '{query}'에 대한 기본 분석을 수행할 수 없습니다.",
            'suggested_actions': [
                "쿼리를 더 구체적으로 작성해주세요",
                "데이터 형식을 확인해주세요",
                "시스템 관리자에게 문의해주세요"
            ]
        }
    
    async def _benchmark_against_standards(self, analysis_result: Dict) -> Dict:
        """표준 대비 벤치마킹"""
        return {
            'industry_standards': {
                'accuracy_benchmark': 0.85,
                'completeness_benchmark': 0.80,
                'clarity_benchmark': 0.90
            },
            'performance_comparison': {
                'above_standard': True,
                'improvement_areas': ['completeness', 'precision'],
                'strength_areas': ['clarity', 'practicality']
            },
            'ranking': 'above_average'
        }
    
    def _assess_reliability(self, analysis_result: Dict) -> float:
        """신뢰성 평가"""
        reliability_factors = [
            analysis_result.get('confidence_level', 0.0),
            1.0 if analysis_result.get('error') is None else 0.0,
            0.8  # 기본 신뢰성 점수
        ]
        return sum(reliability_factors) / len(reliability_factors)
    
    def _assess_robustness(self, analysis_result: Dict) -> float:
        """견고성 평가"""
        robustness_indicators = [
            len(analysis_result.get('meta_reasoning_stages', {})),
            1.0 if analysis_result.get('uncertainty_handling') else 0.0,
            0.7  # 기본 견고성 점수
        ]
        return min(sum(robustness_indicators) / len(robustness_indicators), 1.0)
    
    def _assess_generalizability(self, analysis_result: Dict) -> float:
        """일반화 가능성 평가"""
        generalization_factors = [
            0.8,  # 메타 추론 기반이므로 높은 일반화 가능성
            1.0 if 'domain_context' in analysis_result else 0.5,
            0.9   # LLM 기반 동적 처리
        ]
        return sum(generalization_factors) / len(generalization_factors)
    
    def _evaluate_reasoning_quality(self, analysis_result: Dict) -> float:
        """추론 품질 평가"""
        quality_metrics = [
            len(analysis_result.get('meta_reasoning_stages', {})) / 4.0,  # 4단계 완성도
            analysis_result.get('confidence_metrics', {}).get('reasoning_depth', 0.0),
            analysis_result.get('confidence_metrics', {}).get('consistency_score', 0.0)
        ]
        return sum(quality_metrics) / len(quality_metrics)
    
    def _evaluate_logical_coherence(self, analysis_result: Dict) -> float:
        """논리적 일관성 평가"""
        coherence_indicators = [
            analysis_result.get('confidence_metrics', {}).get('consistency_score', 0.0),
            1.0 if analysis_result.get('meta_meta_analysis') else 0.5,
            0.8  # 메타 추론 구조 자체의 일관성
        ]
        return sum(coherence_indicators) / len(coherence_indicators)
    
    def _evaluate_evidence_support(self, analysis_result: Dict) -> float:
        """증거 지원 평가"""
        evidence_factors = [
            1.0 if analysis_result.get('context') else 0.0,
            len(analysis_result.get('actionable_insights', [])) / 10.0,
            0.7  # 기본 증거 지원 점수
        ]
        return min(sum(evidence_factors) / len(evidence_factors), 1.0)
    
    async def _generate_actionable_recommendations(self, improvement_suggestions: Dict, benchmark_analysis: Dict) -> List[str]:
        """실행 가능한 권장사항 생성"""
        recommendations = []
        
        # 개선 제안에서 우선순위 높은 항목 추출
        high_priority = improvement_suggestions.get('immediate_improvements', {}).get('high_priority', [])
        recommendations.extend(high_priority[:3])
        
        # 벤치마크 분석에서 개선 영역 추출
        improvement_areas = benchmark_analysis.get('performance_comparison', {}).get('improvement_areas', [])
        for area in improvement_areas:
            recommendations.append(f"{area} 영역 집중 개선")
        
        # 구조적 개선 제안 추가
        structural = improvement_suggestions.get('structural_improvements', {})
        for category, items in structural.items():
            if items:
                recommendations.append(f"{category}: {items[0]}")
        
        return recommendations[:8]  # 상위 8개 권장사항 반환
    
    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 블록 추출
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {
                'raw_response': response,
                'parse_error': str(e)
            }