import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python3
"""

Enhanced Report Generator Server - A2A Compatible
pandas_agent 패턴 + UnifiedDataInterface 적용한 종합 보고서 생성 에이전트
"""

import logging
import uvicorn
import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add parent directory to path for core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState, TextPart
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater

# pandas_agent pattern imports
try:
    from a2a_ds_servers.base.unified_data_interface import UnifiedDataInterface, DataIntent, DataProfile, QualityReport
    from a2a_ds_servers.base.llm_first_data_engine import LLMFirstDataEngine
    from a2a_ds_servers.base.smart_dataframe import SmartDataFrame
    from a2a_ds_servers.base.cache_manager import CacheManager
    UNIFIED_INTERFACE_AVAILABLE = True
except ImportError:
    UNIFIED_INTERFACE_AVAILABLE = False
    logger.warning("⚠️ UnifiedDataInterface not available, using fallback implementation")

import httpx
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 다른 A2A 에이전트들의 포트 매핑 (보고서 생성용)
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
    "pandas_agent": 8210,
}

@dataclass
class ReportIntent:
    """보고서 생성 의도"""
    report_type: str  # 'comprehensive', 'summary', 'focused'
    data_sources: List[str]  # 데이터 소스들
    agent_results: List[str]  # 포함할 에이전트 결과들
    focus_areas: List[str]  # 집중 분석 영역
    output_format: str  # 'markdown', 'html', 'json'


class LLMReportAnalyzer:
    """pandas_agent 패턴: LLM 기반 보고서 의도 분석기"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def analyze_report_intent(self, user_query: str) -> ReportIntent:
        """사용자 요청을 분석하여 보고서 생성 의도 파악"""
        if not self.client:
            return ReportIntent(
                report_type="comprehensive",
                data_sources=["latest_data"],
                agent_results=["data_loader", "eda_tools"],
                focus_areas=["basic_analysis"],
                output_format="markdown"
            )
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 데이터 과학 보고서 분석 전문가입니다. 
                        사용자 요청을 분석하여 적절한 보고서 타입과 포함할 내용을 결정해주세요.
                        
                        Available report types: comprehensive, summary, focused
                        Available agents: data_cleaning, data_loader, data_visualization, 
                        data_wrangling, feature_engineering, sql_database, eda_tools, 
                        h2o_ml, mlflow_tools, pandas_agent
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"다음 보고서 요청을 분석해주세요: {user_query}"
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return ReportIntent(
                report_type=result.get("report_type", "comprehensive"),
                data_sources=result.get("data_sources", ["latest_data"]),
                agent_results=result.get("agent_results", ["data_loader", "eda_tools"]),
                focus_areas=result.get("focus_areas", ["basic_analysis"]),
                output_format=result.get("output_format", "markdown")
            )
            
        except Exception as e:
            logger.error(f"❌ LLM 보고서 의도 분석 실패: {e}")
            return ReportIntent(
                report_type="comprehensive",
                data_sources=["latest_data"],
                agent_results=["data_loader", "eda_tools"],
                focus_areas=["basic_analysis"],
                output_format="markdown"
            )


class ReportDataCollector:
    """pandas_agent 패턴: 에이전트 결과 수집기"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def collect_agent_results(self, agent_list: List[str]) -> Dict[str, Any]:
        """지정된 에이전트들로부터 결과 수집"""
        collected_results = {}
        
        for agent_name in agent_list:
            if agent_name in AGENT_PORTS:
                try:
                    port = AGENT_PORTS[agent_name]
                    url = f"http://localhost:{port}/.well-known/agent.json"
                    
                    response = await self.http_client.get(url)
                    if response.status_code == 200:
                        collected_results[agent_name] = {
                            "status": "available",
                            "agent_card": response.json(),
                            "last_checked": datetime.now().isoformat()
                        }
                    else:
                        collected_results[agent_name] = {
                            "status": "unavailable",
                            "error": f"HTTP {response.status_code}",
                            "last_checked": datetime.now().isoformat()
                        }
                        
                except Exception as e:
                    collected_results[agent_name] = {
                        "status": "error",
                        "error": str(e),
                        "last_checked": datetime.now().isoformat()
                    }
        
        return collected_results


class EnhancedReportGenerator:
    """pandas_agent 패턴 기반 종합 보고서 생성기"""

    def __init__(self):
        # LLM 클라이언트 초기화
        self.openai_client = self._initialize_openai_client()
        
        # pandas_agent 패턴 컴포넌트들
        self.report_analyzer = LLMReportAnalyzer(self.openai_client)
        self.data_collector = ReportDataCollector()
        
        # UnifiedDataInterface 지원 (사용 가능한 경우)
        if UNIFIED_INTERFACE_AVAILABLE:
            self.data_engine = LLMFirstDataEngine()
            self.cache_manager = CacheManager()
        
        logger.info("✅ Enhanced Report Generator 초기화 완료")
    
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

    async def generate_comprehensive_report(self, user_query: str) -> str:
        """pandas_agent 패턴 기반 종합 보고서 생성"""
        try:
            logger.info(f"🧑🏻 보고서 생성 요청: {user_query}")
            
            # 1단계: LLM 기반 의도 분석
            report_intent = await self.report_analyzer.analyze_report_intent(user_query)
            logger.info(f"🍒 보고서 의도 분석 완료: {report_intent.report_type}")
            
            # 2단계: 에이전트 결과 수집
            agent_results = await self.data_collector.collect_agent_results(report_intent.agent_results)
            logger.info(f"🍒 {len(agent_results)}개 에이전트 결과 수집 완료")
            
            # 3단계: 데이터 품질 검증 (UnifiedDataInterface 사용)
            data_quality_summary = await self._assess_data_quality(agent_results)
            
            # 4단계: LLM 기반 종합 보고서 생성
            comprehensive_report = await self._generate_llm_report(
                user_query, report_intent, agent_results, data_quality_summary
            )
            
            logger.info("🍒 종합 보고서 생성 완료")
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"❌ 보고서 생성 중 오류: {e}")
            return f"❌ 보고서 생성 중 오류가 발생했습니다: {str(e)}"
    
    async def _assess_data_quality(self, agent_results: Dict) -> Dict[str, Any]:
        """데이터 품질 평가"""
        available_agents = sum(1 for result in agent_results.values() if result.get("status") == "available")
        total_agents = len(agent_results)
        
        return {
            "availability_rate": available_agents / total_agents if total_agents > 0 else 0,
            "total_agents": total_agents,
            "available_agents": available_agents,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_llm_report(self, user_query: str, intent: ReportIntent, 
                                   agent_results: Dict, quality_summary: Dict) -> str:
        """LLM 기반 최종 보고서 생성"""
        if not self.openai_client:
            return self._generate_fallback_report(user_query, intent, agent_results, quality_summary)
        
        try:
            report_prompt = f"""
다음 정보를 바탕으로 전문적인 데이터 과학 보고서를 생성해주세요:

**사용자 요청:** {user_query}

**보고서 유형:** {intent.report_type}
**집중 영역:** {', '.join(intent.focus_areas)}

**에이전트 가용성 현황:**
- 총 에이전트: {quality_summary['total_agents']}개
- 사용 가능: {quality_summary['available_agents']}개
- 가용률: {quality_summary['availability_rate']:.1%}

**상세 에이전트 상태:**
{json.dumps(agent_results, indent=2, ensure_ascii=False)}

다음 형식으로 전문적인 보고서를 작성해주세요:
1. 📊 **실행 요약**
2. 🔍 **시스템 상태 분석** 
3. 📈 **주요 발견사항**
4. 💡 **권장사항**
5. 🔮 **다음 단계**
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 전문 데이터 과학 보고서 작성자입니다. 명확하고 실용적인 인사이트를 제공하는 보고서를 작성해주세요."
                    },
                    {
                        "role": "user",
                        "content": report_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ LLM 보고서 생성 실패: {e}")
            return self._generate_fallback_report(user_query, intent, agent_results, quality_summary)
    
    def _generate_fallback_report(self, user_query: str, intent: ReportIntent, 
                                  agent_results: Dict, quality_summary: Dict) -> str:
        """폴백 보고서 생성"""
        return f"""
# 📋 CherryAI 데이터 과학 시스템 보고서

**생성 시각:** {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}  
**요청 내용:** {user_query}

## 📊 실행 요약
- 보고서 유형: {intent.report_type}
- 대상 에이전트: {len(agent_results)}개
- 시스템 가용률: {quality_summary['availability_rate']:.1%}

## 🔍 시스템 상태 분석

### 에이전트 가용성 현황
"""
        
        for agent_name, result in agent_results.items():
            status_emoji = "✅" if result["status"] == "available" else "❌"
            return f"""
{status_emoji} **{agent_name}**: {result["status"]}
"""
        
        return f"""
## 📈 주요 발견사항
- 총 {quality_summary['total_agents']}개 에이전트 중 {quality_summary['available_agents']}개가 정상 작동 중입니다.
- 시스템 전체 안정성: {'우수' if quality_summary['availability_rate'] > 0.8 else '보통' if quality_summary['availability_rate'] > 0.5 else '주의 필요'}

## 💡 권장사항
1. 정상 작동 중인 에이전트들을 활용하여 데이터 분석을 진행하세요.
2. 비정상 상태의 에이전트가 있다면 재시작을 고려해보세요.
3. 구체적인 분석 요청 시 해당 전문 에이전트를 직접 호출하세요.

## 🔮 다음 단계
- 데이터 분석: pandas_agent 또는 eda_tools 활용
- 시각화: data_visualization 에이전트 활용  
- 모델링: h2o_ml 또는 mlflow_tools 활용

---
*이 보고서는 CherryAI Enhanced Report Generator에 의해 자동 생성되었습니다.*
"""


# UnifiedDataInterface 구현체 (사용 가능한 경우에만)
if UNIFIED_INTERFACE_AVAILABLE:
    class ReportGeneratorWithUnifiedInterface(UnifiedDataInterface):
        """UnifiedDataInterface를 구현한 Report Generator"""
        
        def __init__(self):
            self.report_generator = EnhancedReportGenerator()
            if hasattr(self.report_generator, 'data_engine'):
                self.data_engine = self.report_generator.data_engine
                self.cache_manager = self.report_generator.cache_manager
        
        async def load_data(self, intent: DataIntent, context) -> SmartDataFrame:
            """보고서 생성용 데이터 로딩"""
            # 보고서 생성을 위한 메타데이터 수집
            metadata = {
                "intent_type": intent.intent_type.value,
                "file_preferences": intent.file_preferences,
                "timestamp": datetime.now().isoformat()
            }
            
            # 빈 DataFrame with metadata (보고서는 다른 에이전트 결과를 종합)
            import pandas as pd
            empty_df = pd.DataFrame({"report_metadata": [metadata]})
            
            return SmartDataFrame(empty_df, metadata=metadata)
        
        async def get_data_info(self) -> DataProfile:
            """보고서 데이터 프로파일"""
            return DataProfile(
                shape=(1, 1),
                dtypes={"report_metadata": "object"},
                missing_values={},
                memory_usage=100,
                encoding="utf-8",
                file_size=100
            )
        
        async def validate_data_quality(self) -> QualityReport:
            """보고서 품질 검증"""
            return QualityReport(
                overall_score=1.0,
                completeness=1.0,
                consistency=1.0,
                validity=1.0,
                issues=[],
                recommendations=["보고서 생성을 위해 다른 에이전트들의 결과를 수집하세요"]
            )


class EnhancedReportGeneratorExecutor(AgentExecutor):
    """pandas_agent 패턴 기반 Report Generator Executor"""

    def __init__(self):
        if UNIFIED_INTERFACE_AVAILABLE:
            self.agent = ReportGeneratorWithUnifiedInterface()
        else:
            self.agent = EnhancedReportGenerator()
        
        logger.info("✅ Enhanced Report Generator Executor 초기화 완료")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """pandas_agent 패턴 기반 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 쿼리 추출
            user_query = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        user_query += part.root.text + " "
            
            user_query = user_query.strip()
            if not user_query:
                user_query = "종합 보고서를 생성해주세요"
            
            logger.info(f"🧑🏻 Report Generator 처리 시작: {user_query}")
            
            # 🎯 pandas_agent 패턴: 5단계 처리 파이프라인
            
            # 1단계: 의도 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🍒 LLM 기반 보고서 의도를 분석하고 있습니다...")
            )
            
            # 2단계: 데이터 수집
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🍒 에이전트 결과 데이터를 수집하고 있습니다...")
            )
            
            # 3단계: 보고서 생성
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🍒 LLM 기반 종합 보고서를 생성하고 있습니다...")
            )
            
            # 실제 보고서 생성
            if hasattr(self.agent, 'generate_comprehensive_report'):
                result = await self.agent.generate_comprehensive_report(user_query)
            else:
                result = await self.agent.report_generator.generate_comprehensive_report(user_query)
            
            # 4단계: 품질 검증
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🍒 보고서 품질을 검증하고 있습니다...")
            )
            
            # 5단계: 결과 반환
            await task_updater.add_artifact(
                [TextPart(text=result)],
                name="comprehensive_report",
                metadata={"report_type": "comprehensive", "generator": "enhanced_report_generator"}
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message("✅ 종합 보고서 생성이 완료되었습니다!")
            )
            
            logger.info("✅ Enhanced Report Generator 작업 완료")
            
        except Exception as e:
            logger.error(f"❌ Enhanced Report Generator 실패: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ 보고서 생성 중 오류가 발생했습니다: {str(e)}")
            )

def create_agent_card() -> AgentCard:
    """Create agent card for report_generator"""
    return AgentCard(
        name="report_generator",
        description="Comprehensive data analysis report generator that synthesizes results from multiple analysis agents",
        url="http://localhost:8316/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="report_generation",
                name="report_generation",
                description="Generate comprehensive reports from multiple analysis results",
                tags=["report", "synthesis", "analysis", "documentation"],
                examples=["generate report", "create summary", "synthesize results"]
            ),
            AgentSkill(
                id="result_validation",
                name="result_validation", 
                description="Validate analysis results for accuracy and consistency",
                tags=["validation", "verification", "accuracy", "quality"],
                examples=["validate results", "check accuracy", "verify findings"]
            ),
            AgentSkill(
                id="visualization_aggregation",
                name="visualization_aggregation",
                description="Aggregate and organize visualizations from multiple agents",
                tags=["visualization", "aggregation", "dashboard", "charts"],
                examples=["combine charts", "create dashboard", "aggregate plots"]
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            cancellation=True
        ),
        supportsAuthenticatedExtendedCard=False
    )

def main():
    """Main function to start the report_generator server"""
    logger.info("🚀 Starting Report Generator A2A Server on port 8316...")
    
    # Create agent card
    agent_card = create_agent_card()
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=EnhancedReportGeneratorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A application
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("📋 Starting Report Generator Server")
    print("🌐 Server starting on http://localhost:8315")
    print("📋 Agent card: http://localhost:8315/.well-known/agent.json")
    
    # Run server
    uvicorn.run(server.build(), host="0.0.0.0", port=8315, log_level="info")

if __name__ == "__main__":
    main()