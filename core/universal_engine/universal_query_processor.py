"""
Universal Query Processor - 완전 범용 쿼리 처리기

요구사항 1에 따른 Zero Hardcoding Architecture 구현
- 어떤 도메인 가정도 하지 않음
- 순수 LLM 기반 동적 분석
- 모든 처리 과정을 메타 추론으로 결정
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..llm_factory import LLMFactory
from .meta_reasoning_engine import MetaReasoningEngine
from .dynamic_knowledge_orchestrator import DynamicKnowledgeOrchestrator
from .adaptive_response_generator import AdaptiveResponseGenerator
from .real_time_learning_system import RealTimeLearningSystem

logger = logging.getLogger(__name__)


class UniversalQueryProcessor:
    """
    완전 범용 쿼리 처리기
    - 어떤 도메인 가정도 하지 않음
    - 순수 LLM 기반 동적 분석
    - 모든 처리 과정을 메타 추론으로 결정
    """
    
    def __init__(self):
        """UniversalQueryProcessor 초기화"""
        self.llm_client = self._initialize_llm()
        self.meta_reasoning_engine = MetaReasoningEngine()
        self.knowledge_orchestrator = DynamicKnowledgeOrchestrator()
        self.response_generator = AdaptiveResponseGenerator()
        self.learning_system = RealTimeLearningSystem()
        
        logger.info("UniversalQueryProcessor initialized with zero hardcoding")
        
    def _initialize_llm(self):
        """LLM 클라이언트 초기화"""
        try:
            return LLMFactory.create_llm()
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def process_query(self, query: str, data: Any, context: Dict = None) -> Dict:
        """
        완전 동적 쿼리 처리 파이프라인
        
        요구사항 1.4에 따른 구현:
        1. 메타 추론으로 처리 전략 결정
        2. 동적 지식 통합 및 분석
        3. 적응형 응답 생성
        4. 실시간 학습 및 개선
        
        Args:
            query: 사용자 쿼리
            data: 분석 대상 데이터
            context: 추가 컨텍스트 정보
            
        Returns:
            처리 결과 딕셔너리
        """
        logger.info(f"Processing query with Universal Engine: {query[:100]}...")
        
        # 컨텍스트 초기화
        if context is None:
            context = {}
            
        try:
            # 1. 메타 추론으로 전체 처리 전략 결정
            logger.debug("Starting meta-reasoning analysis")
            meta_analysis = await self.meta_reasoning_engine.analyze_request(
                query=query, 
                data=data, 
                context=context
            )
            
            # 2. 동적 지식 통합 및 추론
            logger.debug("Performing dynamic knowledge orchestration")
            knowledge_result = await self.knowledge_orchestrator.process_with_context(
                meta_analysis=meta_analysis,
                query=query,
                data=data
            )
            
            # 3. 적응형 응답 생성
            logger.debug("Generating adaptive response")
            response = await self.response_generator.generate_adaptive_response(
                knowledge_result=knowledge_result,
                user_profile=meta_analysis.get('user_profile', {}),
                interaction_context=context
            )
            
            # 4. 실시간 학습
            logger.debug("Learning from interaction")
            await self.learning_system.learn_from_interaction({
                'query': query,
                'data_characteristics': meta_analysis.get('data_characteristics'),
                'response': response,
                'user_profile': meta_analysis.get('user_profile'),
                'timestamp': datetime.now()
            })
            
            # 결과 반환
            return {
                'success': True,
                'response': response,
                'meta_analysis': meta_analysis,
                'knowledge_result': knowledge_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def process_with_streaming(self, query: str, data: Any, context: Dict = None):
        """
        스트리밍 방식의 쿼리 처리
        
        Args:
            query: 사용자 쿼리
            data: 분석 대상 데이터
            context: 추가 컨텍스트 정보
            
        Yields:
            처리 과정의 스트리밍 결과
        """
        logger.info(f"Processing query with streaming: {query[:100]}...")
        
        # 초기 상태 전송
        yield {
            'type': 'status',
            'message': '메타 추론 시작...',
            'timestamp': datetime.now().isoformat()
        }
        
        # 메타 추론 수행
        meta_analysis = await self.meta_reasoning_engine.analyze_request(
            query=query,
            data=data,
            context=context or {}
        )
        
        yield {
            'type': 'meta_analysis',
            'data': meta_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # 지식 통합
        yield {
            'type': 'status',
            'message': '동적 지식 통합 중...',
            'timestamp': datetime.now().isoformat()
        }
        
        knowledge_result = await self.knowledge_orchestrator.process_with_context(
            meta_analysis=meta_analysis,
            query=query,
            data=data
        )
        
        yield {
            'type': 'knowledge_result',
            'data': knowledge_result,
            'timestamp': datetime.now().isoformat()
        }
        
        # 응답 생성
        yield {
            'type': 'status',
            'message': '적응형 응답 생성 중...',
            'timestamp': datetime.now().isoformat()
        }
        
        response = await self.response_generator.generate_adaptive_response(
            knowledge_result=knowledge_result,
            user_profile=meta_analysis.get('user_profile', {}),
            interaction_context=context or {}
        )
        
        yield {
            'type': 'final_response',
            'data': response,
            'timestamp': datetime.now().isoformat()
        }
        
        # 학습 수행
        await self.learning_system.learn_from_interaction({
            'query': query,
            'data_characteristics': meta_analysis.get('data_characteristics'),
            'response': response,
            'user_profile': meta_analysis.get('user_profile'),
            'timestamp': datetime.now()
        })
        
        yield {
            'type': 'complete',
            'message': '처리 완료',
            'timestamp': datetime.now().isoformat()
        }
    
    async def handle_clarification(self, clarification: str, previous_context: Dict) -> Dict:
        """
        사용자의 명확화 응답 처리
        
        Args:
            clarification: 사용자의 명확화 응답
            previous_context: 이전 대화 컨텍스트
            
        Returns:
            업데이트된 처리 결과
        """
        logger.info("Handling user clarification")
        
        # 이전 컨텍스트와 함께 재처리
        updated_context = {
            **previous_context,
            'clarification': clarification,
            'clarification_timestamp': datetime.now().isoformat()
        }
        
        # 원래 쿼리와 데이터로 재처리
        return await self.process_query(
            query=previous_context.get('original_query', ''),
            data=previous_context.get('original_data'),
            context=updated_context
        )
    
    def get_system_status(self) -> Dict:
        """
        시스템 상태 조회
        
        Returns:
            시스템 상태 정보
        """
        return {
            'status': 'operational',
            'components': {
                'meta_reasoning_engine': 'active',
                'knowledge_orchestrator': 'active',
                'response_generator': 'active',
                'learning_system': 'active'
            },
            'timestamp': datetime.now().isoformat()
        }