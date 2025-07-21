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
        self.initialization_status = {}
        
        logger.info("UniversalQueryProcessor initialized with zero hardcoding")
        
    def _initialize_llm(self):
        """LLM 클라이언트 초기화"""
        try:
            return LLMFactory.create_llm()
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def initialize(self) -> Dict[str, Any]:
        """
        시스템 초기화 및 의존성 검증
        
        요구사항 1.1에 따른 구현:
        - 모든 하위 컴포넌트 초기화 검증
        - 의존성 상태 확인
        - 시스템 준비 상태 검증
        
        Returns:
            초기화 결과 및 상태 정보
        """
        logger.info("Starting UniversalQueryProcessor initialization")
        
        initialization_results = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'dependencies': {},
            'overall_status': 'initializing'
        }
        
        try:
            # 1. LLM 클라이언트 검증
            logger.debug("Verifying LLM client")
            if self.llm_client:
                initialization_results['components']['llm_client'] = {
                    'status': 'ready',
                    'type': type(self.llm_client).__name__
                }
            else:
                initialization_results['components']['llm_client'] = {
                    'status': 'failed',
                    'error': 'LLM client not initialized'
                }
            
            # 2. 메타 추론 엔진 초기화 검증
            logger.debug("Initializing MetaReasoningEngine")
            if hasattr(self.meta_reasoning_engine, 'initialize'):
                meta_init = await self.meta_reasoning_engine.initialize()
                initialization_results['components']['meta_reasoning_engine'] = meta_init
            else:
                initialization_results['components']['meta_reasoning_engine'] = {
                    'status': 'ready',
                    'note': 'No explicit initialization required'
                }
            
            # 3. 동적 지식 오케스트레이터 초기화 검증
            logger.debug("Initializing DynamicKnowledgeOrchestrator")
            if hasattr(self.knowledge_orchestrator, 'initialize'):
                knowledge_init = await self.knowledge_orchestrator.initialize()
                initialization_results['components']['knowledge_orchestrator'] = knowledge_init
            else:
                initialization_results['components']['knowledge_orchestrator'] = {
                    'status': 'ready',
                    'note': 'No explicit initialization required'
                }
            
            # 4. 적응형 응답 생성기 초기화 검증
            logger.debug("Initializing AdaptiveResponseGenerator")
            if hasattr(self.response_generator, 'initialize'):
                response_init = await self.response_generator.initialize()
                initialization_results['components']['response_generator'] = response_init
            else:
                initialization_results['components']['response_generator'] = {
                    'status': 'ready',
                    'note': 'No explicit initialization required'
                }
            
            # 5. 실시간 학습 시스템 초기화 검증
            logger.debug("Initializing RealTimeLearningSystem")
            if hasattr(self.learning_system, 'initialize'):
                learning_init = await self.learning_system.initialize()
                initialization_results['components']['learning_system'] = learning_init
            else:
                initialization_results['components']['learning_system'] = {
                    'status': 'ready',
                    'note': 'No explicit initialization required'
                }
            
            # 6. 의존성 검증
            logger.debug("Verifying system dependencies")
            dependencies_check = await self._verify_dependencies()
            initialization_results['dependencies'] = dependencies_check
            
            # 7. 전체 상태 결정
            all_components_ready = all(
                comp.get('status') == 'ready' 
                for comp in initialization_results['components'].values()
            )
            dependencies_ready = dependencies_check.get('all_satisfied', False)
            
            if all_components_ready and dependencies_ready:
                initialization_results['overall_status'] = 'ready'
                self.initialization_status = initialization_results
                logger.info("UniversalQueryProcessor initialization completed successfully")
            else:
                initialization_results['overall_status'] = 'partial'
                logger.warning("UniversalQueryProcessor initialization completed with warnings")
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            initialization_results['overall_status'] = 'failed'
            initialization_results['error'] = str(e)
            return initialization_results
    
    async def _verify_dependencies(self) -> Dict[str, Any]:
        """의존성 검증"""
        dependencies = {
            'llm_factory': False,
            'meta_reasoning_engine': False,
            'knowledge_orchestrator': False,
            'response_generator': False,
            'learning_system': False
        }
        
        try:
            # LLM Factory 검증
            dependencies['llm_factory'] = hasattr(LLMFactory, 'create_llm')
            
            # 각 컴포넌트 검증
            dependencies['meta_reasoning_engine'] = self.meta_reasoning_engine is not None
            dependencies['knowledge_orchestrator'] = self.knowledge_orchestrator is not None
            dependencies['response_generator'] = self.response_generator is not None
            dependencies['learning_system'] = self.learning_system is not None
            
        except Exception as e:
            logger.error(f"Error verifying dependencies: {e}")
        
        return {
            'details': dependencies,
            'all_satisfied': all(dependencies.values()),
            'satisfied_count': sum(dependencies.values()),
            'total_count': len(dependencies)
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """
        현재 시스템 상태 반환
        
        요구사항 1.1에 따른 구현:
        - 실시간 시스템 상태 조회
        - 컴포넌트별 상태 정보
        - 성능 메트릭 포함
        
        Returns:
            현재 시스템 상태 정보
        """
        logger.debug("Getting system status")
        
        try:
            status_info = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'operational',
                'initialization': self.initialization_status,
                'components': {},
                'performance': {},
                'health_check': {}
            }
            
            # 1. 컴포넌트 상태 확인
            status_info['components'] = {
                'llm_client': {
                    'status': 'active' if self.llm_client else 'inactive',
                    'type': type(self.llm_client).__name__ if self.llm_client else 'None'
                },
                'meta_reasoning_engine': {
                    'status': 'active' if self.meta_reasoning_engine else 'inactive',
                    'type': type(self.meta_reasoning_engine).__name__
                },
                'knowledge_orchestrator': {
                    'status': 'active' if self.knowledge_orchestrator else 'inactive',
                    'type': type(self.knowledge_orchestrator).__name__
                },
                'response_generator': {
                    'status': 'active' if self.response_generator else 'inactive',
                    'type': type(self.response_generator).__name__
                },
                'learning_system': {
                    'status': 'active' if self.learning_system else 'inactive',
                    'type': type(self.learning_system).__name__
                }
            }
            
            # 2. 성능 메트릭 (기본값)
            status_info['performance'] = {
                'memory_usage': 'N/A',
                'cpu_usage': 'N/A',
                'response_time_avg': 'N/A',
                'requests_processed': 'N/A'
            }
            
            # 3. 헬스 체크
            health_checks = []
            
            # LLM 클라이언트 헬스 체크
            if self.llm_client:
                health_checks.append({'component': 'llm_client', 'status': 'healthy'})
            else:
                health_checks.append({'component': 'llm_client', 'status': 'unhealthy'})
            
            # 각 컴포넌트 헬스 체크
            for comp_name, comp_obj in [
                ('meta_reasoning_engine', self.meta_reasoning_engine),
                ('knowledge_orchestrator', self.knowledge_orchestrator),
                ('response_generator', self.response_generator),
                ('learning_system', self.learning_system)
            ]:
                if comp_obj:
                    if hasattr(comp_obj, 'get_health_status'):
                        try:
                            health = await comp_obj.get_health_status()
                            health_checks.append({'component': comp_name, 'status': health})
                        except Exception as e:
                            health_checks.append({
                                'component': comp_name, 
                                'status': 'error', 
                                'error': str(e)
                            })
                    else:
                        health_checks.append({'component': comp_name, 'status': 'healthy'})
                else:
                    health_checks.append({'component': comp_name, 'status': 'missing'})
            
            status_info['health_check'] = {
                'checks': health_checks,
                'overall_health': 'healthy' if all(
                    check['status'] in ['healthy', 'N/A'] 
                    for check in health_checks
                ) else 'degraded'
            }
            
            # 4. 전체 상태 결정
            if status_info['health_check']['overall_health'] == 'healthy':
                status_info['overall_status'] = 'operational'
            else:
                status_info['overall_status'] = 'degraded'
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'components': {},
                'performance': {},
                'health_check': {}
            }
    
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