import json
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """컨텍스트 타입 분류"""
    DOMAIN_EXPERT = "domain_expert"  # 도메인 전문가 역할
    TECHNICAL_SPECIALIST = "technical_specialist"  # 기술 전문가
    BUSINESS_ANALYST = "business_analyst"  # 비즈니스 분석가
    GENERAL_ASSISTANT = "general_assistant"  # 일반 어시스턴트
    CUSTOM_ROLE = "custom_role"  # 사용자 정의 역할

@dataclass
class PromptContext:
    """프롬프트 컨텍스트 구조"""
    context_type: ContextType
    role_description: str
    domain_knowledge: str
    task_requirements: str
    data_context: str
    original_prompt: str
    confidence_score: float

class IntelligentPromptHandler:
    """LLM 기반 지능형 프롬프트 컨텍스트 보존 시스템"""
    
    def __init__(self, llm_instance=None):
        """
        Args:
            llm_instance: LLM 인스턴스 (없으면 기본 LLM 사용)
        """
        if llm_instance:
            self.llm = llm_instance
        else:
            from core.llm_factory import create_llm_instance
            self.llm = create_llm_instance()
        
        logger.info("IntelligentPromptHandler initialized with LLM-based context analysis")
    
    async def analyze_prompt_context(self, user_request: str, data_info: str = "") -> PromptContext:
        """
        LLM을 사용하여 프롬프트 컨텍스트를 지능적으로 분석
        
        Args:
            user_request: 사용자의 원본 요청
            data_info: 데이터 정보 (파일명, 형태 등)
            
        Returns:
            PromptContext: 분석된 컨텍스트 정보
        """
        
        analysis_prompt = f"""
당신은 AI 프롬프트 분석 전문가입니다. 주어진 사용자 요청을 분석하여 다음 정보를 JSON 형태로 추출해주세요.

사용자 요청:
```
{user_request}
```

데이터 정보: {data_info}

다음 형식으로 분석 결과를 제공해주세요:

{{
    "context_type": "domain_expert|technical_specialist|business_analyst|general_assistant|custom_role",
    "role_description": "사용자가 요구하는 역할이나 관점 (예: '20년 경력의 반도체 이온주입 공정 엔지니어', '데이터 사이언티스트', '비즈니스 분석가' 등)",
    "domain_knowledge": "해당 역할에 필요한 도메인 지식이나 전문성 요구사항",
    "task_requirements": "구체적으로 수행해야 할 작업이나 분석 요구사항",
    "data_context": "데이터와 관련된 특별한 요구사항이나 컨텍스트",
    "key_constraints": "중요한 제약사항이나 고려사항",
    "output_format": "원하는 결과 형태나 포맷",
    "confidence_score": 0.0-1.0 (분석 신뢰도)
}}

분석 기준:
1. 역할 지정이 명확한가? (예: "당신은 ~입니다", "~로서", "~의 관점에서")
2. 도메인 전문성이 요구되는가?
3. 특정 업무나 기술 영역의 지식이 필요한가?
4. 데이터 분석의 목적이나 방향성이 명확한가?
5. 결과물에 대한 특별한 요구사항이 있는가?

JSON만 반환하세요:
"""

        try:
            # LLM을 통한 컨텍스트 분석
            response = await self._call_llm_async(analysis_prompt)
            
            # JSON 파싱
            analysis_result = self._extract_json_from_response(response)
            
            # PromptContext 객체 생성
            context = PromptContext(
                context_type=ContextType(analysis_result.get("context_type", "general_assistant")),
                role_description=analysis_result.get("role_description", ""),
                domain_knowledge=analysis_result.get("domain_knowledge", ""),
                task_requirements=analysis_result.get("task_requirements", user_request),
                data_context=analysis_result.get("data_context", data_info),
                original_prompt=user_request,
                confidence_score=float(analysis_result.get("confidence_score", 0.7))
            )
            
            logger.info(f"✅ Prompt context analyzed - Type: {context.context_type.value}, Confidence: {context.confidence_score}")
            return context
            
        except Exception as e:
            logger.warning(f"⚠️ LLM context analysis failed: {e}, using fallback")
            return self._create_fallback_context(user_request, data_info)
    
    async def create_enhanced_prompt(self, context: PromptContext, agent_type: str = "data_analysis") -> str:
        """
        분석된 컨텍스트를 기반으로 향상된 프롬프트 생성
        
        Args:
            context: 분석된 프롬프트 컨텍스트
            agent_type: 에이전트 타입 (data_analysis, visualization, etc.)
            
        Returns:
            str: 향상된 프롬프트
        """
        
        enhancement_prompt = f"""
당신은 AI 프롬프트 최적화 전문가입니다. 분석된 컨텍스트를 바탕으로 {agent_type} 에이전트가 최적의 성능을 발휘할 수 있는 프롬프트를 생성해주세요.

원본 요청: {context.original_prompt}

분석된 컨텍스트:
- 역할 타입: {context.context_type.value}
- 역할 설명: {context.role_description}
- 도메인 지식: {context.domain_knowledge}
- 작업 요구사항: {context.task_requirements}
- 데이터 컨텍스트: {context.data_context}

에이전트 타입: {agent_type}

다음 원칙에 따라 최적화된 프롬프트를 생성해주세요:

1. **역할 보존**: 사용자가 지정한 역할과 관점을 정확히 유지
2. **도메인 전문성**: 해당 분야의 전문 지식과 용어 활용
3. **작업 명확성**: 구체적이고 실행 가능한 작업 지시
4. **데이터 컨텍스트**: 주어진 데이터의 특성과 목적 반영
5. **결과 품질**: 전문적이고 실용적인 결과물 생성

최적화된 프롬프트:
"""

        try:
            enhanced_prompt = await self._call_llm_async(enhancement_prompt)
            
            # 프롬프트 검증
            if self._validate_enhanced_prompt(enhanced_prompt, context):
                logger.info("✅ Enhanced prompt generated successfully")
                return enhanced_prompt.strip()
            else:
                logger.warning("⚠️ Enhanced prompt validation failed, using template")
                return self._create_template_prompt(context, agent_type)
                
        except Exception as e:
            logger.warning(f"⚠️ Enhanced prompt generation failed: {e}, using template")
            return self._create_template_prompt(context, agent_type)
    
    async def _call_llm_async(self, prompt: str) -> str:
        """LLM 비동기 호출"""
        try:
            # LLM 타입에 따른 호출 방식 처리
            if hasattr(self.llm, 'ainvoke'):
                # LangChain 스타일
                response = await self.llm.ainvoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            elif hasattr(self.llm, 'acall'):
                # 일반적인 비동기 호출
                return await self.llm.acall(prompt)
            else:
                # 동기 호출 (fallback)
                import asyncio
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._call_llm_sync, prompt
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _call_llm_sync(self, prompt: str) -> str:
        """LLM 동기 호출 (fallback)"""
        if hasattr(self.llm, 'invoke'):
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        elif hasattr(self.llm, 'call'):
            return self.llm.call(prompt)
        else:
            return self.llm(prompt)
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """응답에서 JSON 추출"""
        try:
            # JSON 블록 찾기
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}")
            # 간단한 파싱 시도
            return self._simple_parse_response(response)
    
    def _simple_parse_response(self, response: str) -> Dict[str, Any]:
        """간단한 응답 파싱 (JSON 실패 시 fallback)"""
        result = {
            "context_type": "general_assistant",
            "role_description": "",
            "domain_knowledge": "",
            "task_requirements": "",
            "data_context": "",
            "key_constraints": "",
            "output_format": "",
            "confidence_score": 0.5
        }
        
        # 키워드 기반 간단 분석
        response_lower = response.lower()
        
        if any(word in response_lower for word in ['엔지니어', '전문가', '분석가', 'engineer', 'expert', 'analyst']):
            result["context_type"] = "domain_expert"
            result["confidence_score"] = 0.6
        
        if any(word in response_lower for word in ['기술', '공정', '시스템', 'technical', 'process', 'system']):
            result["context_type"] = "technical_specialist"
            result["confidence_score"] = 0.7
        
        return result
    
    def _validate_enhanced_prompt(self, prompt: str, context: PromptContext) -> bool:
        """향상된 프롬프트 검증"""
        if not prompt or len(prompt.strip()) < 50:
            return False
        
        # 역할 보존 확인
        if context.role_description and context.role_description not in prompt:
            # 유사한 역할 표현이 있는지 확인
            role_keywords = context.role_description.split()[:3]  # 첫 3단어
            if not any(keyword in prompt for keyword in role_keywords):
                return False
        
        return True
    
    def _create_template_prompt(self, context: PromptContext, agent_type: str) -> str:
        """템플릿 기반 프롬프트 생성 (fallback)"""
        
        base_templates = {
            "data_analysis": """
{role_context}

현재 작업: {task_requirements}

데이터 정보: {data_context}

위의 역할과 전문성을 바탕으로 주어진 데이터에 대해 철저하고 전문적인 분석을 수행해주세요.
분석 결과는 해당 분야의 전문가 수준으로 제공하며, 실무에서 바로 활용할 수 있는 인사이트를 포함해주세요.
""",
            "visualization": """
{role_context}

시각화 요청: {task_requirements}

데이터 정보: {data_context}

해당 분야의 전문가 관점에서 데이터의 핵심 인사이트를 효과적으로 전달할 수 있는 시각화를 생성해주세요.
차트와 그래프는 전문적이면서도 이해하기 쉽게 구성하고, 필요한 설명을 포함해주세요.
""",
            "feature_engineering": """
{role_context}

피처 엔지니어링 요청: {task_requirements}

데이터 정보: {data_context}

도메인 전문 지식을 활용하여 의미 있는 피처를 생성하고, 각 피처의 비즈니스적 의미와 예측력을 설명해주세요.
""",
            "default": """
{role_context}

요청사항: {task_requirements}

데이터 정보: {data_context}

위의 전문성과 역할을 바탕으로 요청된 작업을 수행해주세요.
"""
        }
        
        template = base_templates.get(agent_type, base_templates["default"])
        
        # 역할 컨텍스트 구성
        role_context = ""
        if context.role_description:
            role_context = f"당신은 {context.role_description}입니다."
            if context.domain_knowledge:
                role_context += f"\n\n전문 지식: {context.domain_knowledge}"
        
        return template.format(
            role_context=role_context,
            task_requirements=context.task_requirements,
            data_context=context.data_context
        ).strip()
    
    def _create_fallback_context(self, user_request: str, data_info: str) -> PromptContext:
        """fallback 컨텍스트 생성"""
        return PromptContext(
            context_type=ContextType.GENERAL_ASSISTANT,
            role_description="",
            domain_knowledge="",
            task_requirements=user_request,
            data_context=data_info,
            original_prompt=user_request,
            confidence_score=0.3
        )

    async def process_request_with_context(self, user_request: str, data_info: str, agent_type: str) -> Tuple[str, PromptContext]:
        """
        전체 프로세스: 컨텍스트 분석 → 프롬프트 향상
        
        Returns:
            Tuple[enhanced_prompt, context]: 향상된 프롬프트와 컨텍스트 정보
        """
        # 1. 컨텍스트 분석
        context = await self.analyze_prompt_context(user_request, data_info)
        
        # 2. 프롬프트 향상
        enhanced_prompt = await self.create_enhanced_prompt(context, agent_type)
        
        logger.info(f"🧠 Intelligent prompt processing completed - Confidence: {context.confidence_score}")
        
        return enhanced_prompt, context 