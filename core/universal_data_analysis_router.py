"""
Universal Data Analysis Router

LLM 기반 범용 데이터 분석 라우팅 시스템
사용자의 자연어 질문을 분석하여 가장 적합한 전문 에이전트로 라우팅

Author: CherryAI Team
Date: 2024-12-30
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path

# LLM Integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Enhanced Tracking System
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError:
    ENHANCED_TRACKING_AVAILABLE = False

# UserFileTracker 통합
try:
    from core.user_file_tracker import get_user_file_tracker
    from core.session_data_manager import SessionDataManager
    USER_FILE_TRACKER_AVAILABLE = True
except ImportError:
    USER_FILE_TRACKER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """데이터 분석 유형 분류"""
    PANDAS_AI = "pandas_ai"           # 범용 자연어 데이터 분석
    EDA = "eda"                       # 탐색적 데이터 분석
    VISUALIZATION = "visualization"   # 데이터 시각화
    STATISTICS = "statistics"         # 통계 분석
    MACHINE_LEARNING = "ml"           # 머신러닝
    DATA_CLEANING = "cleaning"        # 데이터 전처리
    DATA_LOADING = "loading"          # 데이터 로딩
    FEATURE_ENGINEERING = "features"  # 피처 엔지니어링
    DATABASE = "database"             # 데이터베이스 쿼리
    GENERAL = "general"               # 일반 질문


@dataclass
class RouteDecision:
    """라우팅 결정 결과"""
    analysis_type: AnalysisType
    confidence: float
    reasoning: str
    recommended_agent: str
    parameters: Dict[str, Any] = None
    fallback_agents: List[str] = None


@dataclass
class AgentCapability:
    """에이전트 능력 정의"""
    agent_name: str
    analysis_types: List[AnalysisType]
    strengths: List[str]
    limitations: List[str]
    endpoint: str
    priority: int = 5  # 1-10, 10이 최고 우선순위


class UniversalDataAnalysisRouter:
    """
    범용 데이터 분석 라우팅 시스템
    
    LLM을 활용하여 사용자 질문을 분석하고 최적의 전문 에이전트로 라우팅
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.openai_client = None
        self.enhanced_tracer = None
        self.user_file_tracker = None
        self.session_data_manager = None
        
        # 에이전트 능력 정의
        self.agent_capabilities = self._initialize_agent_capabilities()
        
        # 라우팅 히스토리
        self.routing_history: List[Dict] = []
        
        # LLM 초기화
        self._initialize_llm()
        
        # Enhanced Tracking 초기화
        if ENHANCED_TRACKING_AVAILABLE:
            try:
                self.enhanced_tracer = get_enhanced_tracer()
                logger.info("✅ Enhanced Langfuse Tracking 활성화")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced Tracking 초기화 실패: {e}")
        
        # UserFileTracker 초기화
        if USER_FILE_TRACKER_AVAILABLE:
            try:
                self.user_file_tracker = get_user_file_tracker()
                self.session_data_manager = SessionDataManager()
                logger.info("✅ UserFileTracker 통합 활성화")
            except Exception as e:
                logger.warning(f"⚠️ UserFileTracker 초기화 실패: {e}")
    
    def _initialize_llm(self):
        """LLM 클라이언트 초기화"""
        if not OPENAI_AVAILABLE:
            logger.warning("⚠️ OpenAI 라이브러리가 설치되지 않음")
            return
        
        try:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음")
                return
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("✅ OpenAI 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
    
    def _initialize_agent_capabilities(self) -> Dict[str, AgentCapability]:
        """에이전트 능력 정의 초기화"""
        return {
            "pandas_ai": AgentCapability(
                agent_name="Universal Pandas-AI Agent",
                analysis_types=[
                    AnalysisType.PANDAS_AI,
                    AnalysisType.GENERAL,
                    AnalysisType.STATISTICS
                ],
                strengths=[
                    "자연어 데이터 질문 처리",
                    "복잡한 데이터 조작",
                    "코드 자동 생성",
                    "멀티턴 대화"
                ],
                limitations=[
                    "시각화 품질 한계",
                    "고급 ML 알고리즘 부족"
                ],
                endpoint="http://localhost:8000",
                priority=8
            ),
            
            "eda": AgentCapability(
                agent_name="Enhanced EDA Tools Agent",
                analysis_types=[
                    AnalysisType.EDA,
                    AnalysisType.STATISTICS,
                    AnalysisType.VISUALIZATION
                ],
                strengths=[
                    "포괄적 탐색적 데이터 분석",
                    "통계적 인사이트",
                    "데이터 품질 검사",
                    "상관관계 분석"
                ],
                limitations=[
                    "자유형식 자연어 처리 제한"
                ],
                endpoint="http://localhost:8001",
                priority=9
            ),
            
            "visualization": AgentCapability(
                agent_name="Data Visualization Agent",
                analysis_types=[
                    AnalysisType.VISUALIZATION
                ],
                strengths=[
                    "고품질 시각화",
                    "대화형 차트",
                    "다양한 플롯 유형",
                    "맞춤형 스타일링"
                ],
                limitations=[
                    "데이터 분석 기능 제한"
                ],
                endpoint="http://localhost:8002",
                priority=7
            ),
            
            "data_cleaning": AgentCapability(
                agent_name="Data Cleaning Agent",
                analysis_types=[
                    AnalysisType.DATA_CLEANING
                ],
                strengths=[
                    "결측값 처리",
                    "이상값 탐지",
                    "데이터 표준화",
                    "품질 향상"
                ],
                limitations=[
                    "분석 기능 부족"
                ],
                endpoint="http://localhost:8003",
                priority=6
            ),
            
            "feature_engineering": AgentCapability(
                agent_name="Feature Engineering Agent",
                analysis_types=[
                    AnalysisType.FEATURE_ENGINEERING
                ],
                strengths=[
                    "피처 생성",
                    "차원 축소",
                    "피처 선택",
                    "변환 기법"
                ],
                limitations=[
                    "도메인 지식 의존"
                ],
                endpoint="http://localhost:8004",
                priority=6
            ),
            
            "ml": AgentCapability(
                agent_name="Machine Learning Agent",
                analysis_types=[
                    AnalysisType.MACHINE_LEARNING
                ],
                strengths=[
                    "ML 모델 구축",
                    "모델 평가",
                    "하이퍼파라미터 튜닝",
                    "예측 분석"
                ],
                limitations=[
                    "복잡한 데이터 전처리 제한"
                ],
                endpoint="http://localhost:8005",
                priority=7
            )
        }
    
    async def analyze_query_intent(self, user_query: str, context: Optional[Dict] = None) -> RouteDecision:
        """
        사용자 질문의 의도를 분석하여 라우팅 결정
        
        Args:
            user_query: 사용자 질문
            context: 추가 컨텍스트 정보
            
        Returns:
            RouteDecision: 라우팅 결정 결과
        """
        try:
            logger.info(f"🔄 질문 의도 분석 시작: {user_query[:100]}...")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "query_intent_analysis",
                    {"query": user_query, "context": context},
                    "Analyzing user query intent for routing"
                )
            
            # LLM이 사용 가능한 경우 고급 분석
            if self.openai_client:
                decision = await self._llm_based_analysis(user_query, context)
            else:
                # 규칙 기반 분석 (폴백)
                decision = self._rule_based_analysis(user_query, context)
            
            # 라우팅 히스토리 기록
            self.routing_history.append({
                "timestamp": asyncio.get_event_loop().time(),
                "query": user_query,
                "decision": decision,
                "context": context
            })
            
            logger.info(f"✅ 라우팅 결정: {decision.recommended_agent} (신뢰도: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"❌ 질문 의도 분석 실패: {e}")
            # 폴백: pandas-ai로 라우팅
            return RouteDecision(
                analysis_type=AnalysisType.PANDAS_AI,
                confidence=0.3,
                reasoning=f"분석 실패로 인한 기본 라우팅: {str(e)}",
                recommended_agent="pandas_ai",
                fallback_agents=["eda"]
            )
    
    def _normalize_agent_name(self, agent_name: str) -> str:
        """에이전트 이름을 정규화하여 키 매핑"""
        # 에이전트 이름 매핑 테이블
        name_mapping = {
            "Universal Pandas-AI Agent": "pandas_ai",
            "pandas_ai": "pandas_ai",
            "Enhanced EDA Tools Agent": "eda", 
            "eda": "eda",
            "Data Visualization Agent": "visualization",
            "visualization": "visualization",
            "Data Cleaning Agent": "data_cleaning",
            "data_cleaning": "data_cleaning",
            "Feature Engineering Agent": "feature_engineering",
            "feature_engineering": "feature_engineering",
            "Machine Learning Agent": "ml",
            "ml": "ml"
        }
        
        # 정확한 매치 우선
        if agent_name in name_mapping:
            return name_mapping[agent_name]
        
        # 부분 매치 시도
        agent_lower = agent_name.lower()
        for key, value in name_mapping.items():
            if key.lower() in agent_lower or value in agent_lower:
                return value
        
        # 매치되지 않으면 기본값
        return "pandas_ai"
    
    async def _llm_based_analysis(self, user_query: str, context: Optional[Dict] = None) -> RouteDecision:
        """LLM 기반 고급 질문 의도 분석"""
        try:
            # 에이전트 능력 요약
            agent_summary = self._create_agent_summary()
            
            # 프롬프트 구성
            system_prompt = f"""당신은 데이터 분석 전문가이며, 사용자의 질문을 분석하여 가장 적합한 전문 에이전트로 라우팅하는 시스템입니다.

사용 가능한 에이전트들:
{agent_summary}

사용자 질문을 분석하고 다음 JSON 형식으로 응답하세요:
{{
    "analysis_type": "분석 유형 (pandas_ai, eda, visualization, statistics, ml, cleaning, loading, features, database, general 중 하나)",
    "confidence": 0.0-1.0 사이의 신뢰도,
    "reasoning": "라우팅 결정 이유 (한국어)",
    "recommended_agent": "추천 에이전트 이름",
    "parameters": {{"추가 파라미터들"}},
    "fallback_agents": ["대안 에이전트 목록"]
}}

분석 기준:
1. 질문의 핵심 의도 파악
2. 필요한 전문성 수준 판단
3. 에이전트별 강점과 한계 고려
4. 신뢰도는 보수적으로 평가"""

            user_prompt = f"""사용자 질문: "{user_query}"

컨텍스트: {json.dumps(context, ensure_ascii=False, indent=2) if context else "없음"}

위 질문을 분석하여 가장 적합한 에이전트로 라우팅해주세요."""

            response = self.openai_client.chat.completions.create(
                model=self.config.get("llm_model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # 응답 파싱
            response_text = response.choices[0].message.content
            
            # JSON 추출 (```json ... ``` 형태 처리)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            try:
                result = json.loads(response_text)
                
                # 에이전트 이름 정규화
                recommended_agent = result.get("recommended_agent", "pandas_ai")
                normalized_agent = self._normalize_agent_name(recommended_agent)
                
                return RouteDecision(
                    analysis_type=AnalysisType(result.get("analysis_type", "pandas_ai")),
                    confidence=float(result.get("confidence", 0.5)),
                    reasoning=result.get("reasoning", "LLM 분석 결과"),
                    recommended_agent=normalized_agent,
                    parameters=result.get("parameters", {}),
                    fallback_agents=result.get("fallback_agents", ["eda", "pandas_ai"])
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️ LLM 응답 파싱 실패, 규칙 기반으로 폴백: {e}")
                return self._rule_based_analysis(user_query, context)
            
        except Exception as e:
            logger.error(f"❌ LLM 기반 분석 실패: {e}")
            return self._rule_based_analysis(user_query, context)
    
    def _rule_based_analysis(self, user_query: str, context: Optional[Dict] = None) -> RouteDecision:
        """규칙 기반 질문 의도 분석 (폴백)"""
        query_lower = user_query.lower()
        
        # 키워드 기반 매칭
        patterns = {
            AnalysisType.VISUALIZATION: [
                r'그래프|차트|플롯|시각화|그림|도표|plot|chart|graph|visualiz',
                r'히트맵|산점도|막대그래프|선그래프|상자그림|heatmap|scatter|bar|line|box'
            ],
            AnalysisType.EDA: [
                r'탐색|분포|상관관계|기술통계|요약|eda|explore|distribution|correlation|summary|describe',
                r'통계|평균|중앙값|표준편차|분산|statistics|mean|median|std|var'
            ],
            AnalysisType.MACHINE_LEARNING: [
                r'예측|모델|분류|회귀|머신러닝|딥러닝|predict|model|classify|regression|ml|machine learning',
                r'학습|훈련|알고리즘|train|algorithm|fit'
            ],
            AnalysisType.DATA_CLEANING: [
                r'정제|전처리|결측|이상값|중복|정리|clean|preprocess|missing|outlier|duplicate|null'
            ],
            AnalysisType.FEATURE_ENGINEERING: [
                r'피처|특성|변수|차원|feature|variable|dimension|encoding|scaling'
            ],
            AnalysisType.DATABASE: [
                r'sql|쿼리|데이터베이스|조인|select|where|join|database|query'
            ]
        }
        
        # 패턴 매칭 스코어 계산
        scores = {}
        for analysis_type, pattern_list in patterns.items():
            score = 0
            for pattern in pattern_list:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            scores[analysis_type] = score
        
        # 최고 스코어 선택
        if scores and max(scores.values()) > 0:
            best_type = max(scores, key=scores.get)
            confidence = min(0.8, scores[best_type] * 0.2)
        else:
            # 기본값: pandas-ai
            best_type = AnalysisType.PANDAS_AI
            confidence = 0.4
        
        # 에이전트 매핑
        agent_mapping = {
            AnalysisType.PANDAS_AI: "pandas_ai",
            AnalysisType.EDA: "eda",
            AnalysisType.VISUALIZATION: "visualization",
            AnalysisType.STATISTICS: "eda",
            AnalysisType.MACHINE_LEARNING: "ml",
            AnalysisType.DATA_CLEANING: "data_cleaning",
            AnalysisType.FEATURE_ENGINEERING: "feature_engineering",
            AnalysisType.DATABASE: "pandas_ai",
            AnalysisType.GENERAL: "pandas_ai"
        }
        
        recommended_agent = agent_mapping.get(best_type, "pandas_ai")
        
        return RouteDecision(
            analysis_type=best_type,
            confidence=confidence,
            reasoning=f"규칙 기반 키워드 매칭 (스코어: {scores.get(best_type, 0)})",
            recommended_agent=recommended_agent,
            parameters={},
            fallback_agents=["pandas_ai", "eda"]
        )
    
    def _create_agent_summary(self) -> str:
        """에이전트 능력 요약 생성"""
        summary_lines = []
        for agent_id, capability in self.agent_capabilities.items():
            analysis_types = [t.value for t in capability.analysis_types]
            summary_lines.append(
                f"- {capability.agent_name} ({agent_id}): "
                f"분석유형={analysis_types}, "
                f"강점={capability.strengths[:2]}, "
                f"우선순위={capability.priority}"
            )
        return "\n".join(summary_lines)
    
    async def route_query(self, user_query: str, session_id: Optional[str] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        질문을 분석하고 적절한 에이전트로 라우팅
        
        Args:
            user_query: 사용자 질문
            session_id: 세션 ID
            context: 추가 컨텍스트
            
        Returns:
            Dict: 라우팅 결과
        """
        try:
            logger.info(f"🔄 질문 라우팅 시작: {user_query[:100]}...")
            
            # 세션 컨텍스트 수집
            enhanced_context = await self._gather_session_context(session_id, context)
            
            # 의도 분석
            decision = await self.analyze_query_intent(user_query, enhanced_context)
            
            # 에이전트 정보 준비
            agent_info = self.agent_capabilities.get(decision.recommended_agent)
            if not agent_info:
                logger.warning(f"⚠️ 알 수 없는 에이전트: {decision.recommended_agent}")
                agent_info = self.agent_capabilities["pandas_ai"]  # 폴백
            
            result = {
                "success": True,
                "decision": {
                    "analysis_type": decision.analysis_type.value,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "recommended_agent": decision.recommended_agent,
                    "agent_endpoint": agent_info.endpoint,
                    "parameters": decision.parameters or {},
                    "fallback_agents": decision.fallback_agents or []
                },
                "agent_info": {
                    "name": agent_info.agent_name,
                    "strengths": agent_info.strengths,
                    "limitations": agent_info.limitations,
                    "endpoint": agent_info.endpoint,
                    "priority": agent_info.priority
                },
                "context": enhanced_context,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"✅ 라우팅 완료: {decision.recommended_agent}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 질문 라우팅 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": {
                    "recommended_agent": "pandas_ai",
                    "agent_endpoint": "http://localhost:8000",
                    "reasoning": "라우팅 실패로 인한 기본 에이전트 사용"
                }
            }
    
    async def _gather_session_context(self, session_id: Optional[str], context: Optional[Dict]) -> Dict[str, Any]:
        """세션 컨텍스트 수집"""
        enhanced_context = context.copy() if context else {}
        
        if session_id and self.session_data_manager:
            try:
                # 업로드된 파일 정보
                uploaded_files = self.session_data_manager.get_uploaded_files(session_id)
                enhanced_context["uploaded_files"] = uploaded_files
                
                # 파일 유형 분석
                if uploaded_files and self.user_file_tracker:
                    file_analysis = []
                    for file_name in uploaded_files:
                        file_path = self.user_file_tracker.get_best_file(
                            session_id=session_id,
                            query=file_name
                        )
                        if file_path:
                            file_ext = Path(file_path).suffix.lower()
                            file_analysis.append({
                                "name": file_name,
                                "path": file_path,
                                "type": file_ext,
                                "size": Path(file_path).stat().st_size if Path(file_path).exists() else 0
                            })
                    enhanced_context["file_analysis"] = file_analysis
                
            except Exception as e:
                logger.warning(f"⚠️ 세션 컨텍스트 수집 실패: {e}")
        
        return enhanced_context
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """라우팅 통계 조회"""
        if not self.routing_history:
            return {"total_queries": 0, "agent_distribution": {}}
        
        total = len(self.routing_history)
        agent_counts = {}
        analysis_type_counts = {}
        confidence_sum = 0
        
        for entry in self.routing_history:
            decision = entry["decision"]
            agent = decision.recommended_agent
            analysis_type = decision.analysis_type.value
            
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            analysis_type_counts[analysis_type] = analysis_type_counts.get(analysis_type, 0) + 1
            confidence_sum += decision.confidence
        
        return {
            "total_queries": total,
            "agent_distribution": agent_counts,
            "analysis_type_distribution": analysis_type_counts,
            "average_confidence": confidence_sum / total if total > 0 else 0,
            "routing_history_size": len(self.routing_history)
        }
    
    def clear_routing_history(self):
        """라우팅 히스토리 정리"""
        self.routing_history.clear()
        logger.info("✅ 라우팅 히스토리 정리 완료")


# 전역 인스턴스
_router_instance = None


def get_universal_router(config: Optional[Dict] = None) -> UniversalDataAnalysisRouter:
    """범용 데이터 분석 라우터 인스턴스 반환"""
    global _router_instance
    if _router_instance is None:
        _router_instance = UniversalDataAnalysisRouter(config)
    return _router_instance


# CLI 테스트 함수
async def test_router():
    """라우터 테스트"""
    router = get_universal_router()
    
    test_queries = [
        "데이터의 분포를 보여주세요",
        "고객 데이터로 매출 예측 모델을 만들어주세요",
        "결측값을 처리해주세요",
        "상관관계를 시각화해주세요",
        "평균 나이는 얼마인가요?",
    ]
    
    for query in test_queries:
        print(f"\n질문: {query}")
        result = await router.route_query(query)
        if result["success"]:
            decision = result["decision"]
            print(f"에이전트: {decision['recommended_agent']}")
            print(f"신뢰도: {decision['confidence']:.2f}")
            print(f"이유: {decision['reasoning']}")
        else:
            print(f"오류: {result['error']}")
    
    # 통계 출력
    print(f"\n라우팅 통계:")
    stats = router.get_routing_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(test_router()) 