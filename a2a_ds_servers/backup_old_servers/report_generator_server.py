import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python
"""

Report Generator Agent for A2A Data Analysis Platform
A2A SDK v0.2.9 compliant implementation

This agent synthesizes analysis results from multiple data analysis agents
and generates comprehensive, accurate reports without hallucinations.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import httpx
from openai import AsyncOpenAI

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater

# Import core utilities
from core.data_manager import DataManager
from core.session_data_manager import SessionDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agent port mapping for data collection
AGENT_PORTS = {
    "data_loader": 8307,
    "data_visualization": 8308,
    "eda_tools": 8312,
    "pandas_agent": 8210,
    "data_cleaning": 8306,
    "data_wrangling": 8309,
    "feature_engineering": 8310,
}

@dataclass
class AnalysisResult:
    """Structured analysis result from an agent"""
    agent_name: str
    timestamp: str
    status: str
    data: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    confidence_score: float = 1.0
    source_references: List[str] = None

    def __post_init__(self):
        if self.source_references is None:
            self.source_references = []


@dataclass
class ReportSection:
    """A section of the final report"""
    title: str
    content: str
    visualizations: List[Dict[str, Any]]
    code_artifacts: List[Dict[str, Any]]
    data_references: List[str]
    confidence_level: float


class LLMEvidenceValidator:
    """LLM-based evidence validation to prevent hallucinations"""
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm_client = llm_client
    
    async def validate_with_llm(self, report_content: str, source_data: List[AnalysisResult]) -> Dict[str, Any]:
        """LLM validates report content against source data"""
        
        source_data_formatted = self._format_source_data_for_validation(source_data)
        
        validation_prompt = f"""
당신은 데이터 분석 보고서 검증 전문가입니다. 다음 보고서의 모든 수치, 통계, 주장을 원본 데이터와 대조하여 검증하세요.

보고서 내용:
{report_content}

원본 데이터:
{source_data_formatted}

검증 작업:
1. 보고서의 모든 수치가 원본 데이터에서 확인 가능한지 체크
2. 통계적 주장이 데이터로 뒷받침되는지 확인
3. 잘못된 해석이나 과장된 표현이 있는지 점검
4. 원본 데이터에 없는 내용이 추가되었는지 확인

다음 JSON 형식으로 응답하세요:
{{
    "validation_status": "valid|needs_correction|invalid",
    "confidence_score": 0.0-1.0,
    "issues_found": [
        {{
            "type": "numerical|statistical|interpretation|unsupported",
            "description": "구체적인 문제점",
            "severity": "high|medium|low"
        }}
    ],
    "validated_report": "수정된 보고서 내용 (필요시)",
    "validation_notes": "검증 과정에서 발견된 주요 사항들"
}}
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.1,
                max_tokens=3000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return {
                "validation_status": "error",
                "confidence_score": 0.0,
                "issues_found": [{"type": "system", "description": f"Validation error: {str(e)}", "severity": "high"}],
                "validated_report": report_content,
                "validation_notes": "Validation failed due to system error"
            }
    
    def _format_source_data_for_validation(self, source_data: List[AnalysisResult]) -> str:
        """Format source data for LLM validation"""
        formatted_data = []
        
        for result in source_data:
            agent_data = {
                "agent": result.agent_name,
                "timestamp": result.timestamp,
                "data": result.data,
                "confidence": result.confidence_score
            }
            formatted_data.append(f"=== {result.agent_name} ===\n{json.dumps(agent_data, indent=2, ensure_ascii=False)}")
        
        return "\n\n".join(formatted_data)


class VisualizationAggregator:
    """Aggregates and validates visualizations from multiple agents"""
    
    def __init__(self):
        self.supported_types = ["line", "bar", "scatter", "heatmap", "pie", "box", "histogram"]
    
    def aggregate(self, agent_results: List[AnalysisResult]) -> List[Dict[str, Any]]:
        """Aggregate visualizations from multiple agents"""
        all_visualizations = []
        
        for result in agent_results:
            if "visualizations" in result.data:
                for viz in result.data["visualizations"]:
                    # Add source agent information
                    viz["source_agent"] = result.agent_name
                    viz["confidence"] = result.confidence_score
                    all_visualizations.append(viz)
        
        # Remove duplicates and prioritize by confidence
        return self._deduplicate_visualizations(all_visualizations)
    
    def _deduplicate_visualizations(self, visualizations: List[Dict]) -> List[Dict]:
        """Remove duplicate visualizations, keeping highest confidence ones"""
        unique_viz = {}
        
        for viz in visualizations:
            # Create a key based on viz type and data characteristics
            key = f"{viz.get('type', 'unknown')}_{viz.get('title', 'untitled')}"
            
            if key not in unique_viz or viz.get("confidence", 0) > unique_viz[key].get("confidence", 0):
                unique_viz[key] = viz
        
        return list(unique_viz.values())
    
    def create_dashboard_layout(self, visualizations: List[Dict]) -> Dict[str, Any]:
        """Create a dashboard layout for visualizations"""
        layout = {
            "title": "Data Analysis Dashboard",
            "grid": [],
            "summary_charts": [],
            "detailed_charts": []
        }
        
        # Categorize visualizations
        for viz in visualizations:
            if viz.get("type") in ["pie", "bar"] and viz.get("priority") == "high":
                layout["summary_charts"].append(viz)
            else:
                layout["detailed_charts"].append(viz)
        
        return layout


class LLMReportGenerator:
    """LLM-driven report generator that creates user-intent-specific reports"""
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm_client = llm_client
    
    async def analyze_report_intent(self, user_query: str) -> Dict[str, Any]:
        """Analyze user's intent for report generation"""
        
        intent_prompt = f"""
사용자 요청을 분석하여 어떤 종류의 보고서를 원하는지 파악하세요.

사용자 요청: "{user_query}"

다음 JSON 형식으로 응답하세요:
{{
    "main_focus": "사용자의 주요 관심사 (예: 트렌드 분석, 이상치 탐지, 상관관계 분석, 예측, 요약통계 등)",
    "detail_level": "summary|detailed|technical",
    "target_audience": "executive|analyst|developer|general",
    "preferred_format": "narrative|bullet_points|structured|mixed",
    "visualization_preference": "high|medium|low",
    "code_inclusion": "high|medium|low|none",
    "key_questions": ["사용자가 답을 원하는 구체적인 질문들"],
    "analysis_scope": "broad|focused|comparative",
    "report_tone": "formal|casual|technical"
}}
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Intent analysis error: {e}")
            return {
                "main_focus": "general analysis",
                "detail_level": "detailed",
                "target_audience": "general",
                "preferred_format": "mixed",
                "visualization_preference": "medium",
                "code_inclusion": "medium",
                "key_questions": [user_query],
                "analysis_scope": "broad",
                "report_tone": "formal"
            }
    
    async def generate_report(self, user_query: str, intent: Dict[str, Any], agent_results: List[AnalysisResult]) -> str:
        """Generate a customized report based on user intent and agent results"""
        
        # Format agent results for LLM consumption
        formatted_results = self._format_agent_results_for_llm(agent_results)
        
        report_prompt = f"""
당신은 데이터 분석 보고서 작성 전문가입니다. 사용자의 요청과 의도에 맞는 맞춤형 보고서를 작성하세요.

사용자 원본 요청: "{user_query}"

사용자 의도 분석:
{json.dumps(intent, indent=2, ensure_ascii=False)}

분석 결과 데이터:
{formatted_results}

보고서 작성 지침:
1. 사용자의 의도({intent.get('main_focus', 'general analysis')})에 정확히 부합하는 내용으로 구성
2. 상세 수준: {intent.get('detail_level', 'detailed')}
3. 대상 독자: {intent.get('target_audience', 'general')}
4. 형식: {intent.get('preferred_format', 'mixed')}
5. 시각화 선호도: {intent.get('visualization_preference', 'medium')}
6. 코드 포함 수준: {intent.get('code_inclusion', 'medium')}
7. 톤: {intent.get('report_tone', 'formal')}

핵심 답변 질문들:
{chr(10).join(f"- {q}" for q in intent.get('key_questions', [user_query]))}

중요 원칙:
- 모든 수치와 통계는 제공된 분석 결과에서만 인용
- 각 주장에는 반드시 출처 명시 (예: [pandas_agent 분석], [EDA 결과])
- 추측이나 가정은 명시적으로 표현
- 데이터에 없는 내용은 절대 추가하지 않음
- 사용자가 원하는 답변에 집중

보고서를 작성하세요:
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": report_prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _format_agent_results_for_llm(self, agent_results: List[AnalysisResult]) -> str:
        """Format agent results for LLM consumption"""
        formatted_sections = []
        
        for result in agent_results:
            section = f"""
=== {result.agent_name} 분석 결과 ===
실행 시간: {result.timestamp}
신뢰도: {result.confidence_score:.2f}

분석 데이터:
{json.dumps(result.data, indent=2, ensure_ascii=False)}

아티팩트:
{json.dumps(result.artifacts, indent=2, ensure_ascii=False)}

데이터 출처: {', '.join(result.source_references)}
"""
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)


class ReportGeneratorAgent:
    """Main Report Generator Agent that orchestrates LLM-driven report creation"""
    
    def __init__(self):
        # Initialize LLM client
        self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize LLM-driven components
        self.validator = LLMEvidenceValidator(self.llm_client)
        self.visualizer = VisualizationAggregator()
        self.report_generator = LLMReportGenerator(self.llm_client)
        
        # Keep existing data managers
        self.data_manager = DataManager()
        self.session_manager = SessionDataManager()
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def generate_report(self, session_id: str, user_query: str, agent_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive LLM-driven report based on agent analysis results
        
        Args:
            session_id: Session identifier for data retrieval
            user_query: Original user query for context
            agent_list: List of agents to collect results from (None = all)
        
        Returns:
            Generated report with all artifacts
        """
        logger.info(f"🚀 Generating LLM-driven report for session: {session_id}")
        
        # Step 1: Analyze user intent
        intent_analysis = await self.report_generator.analyze_report_intent(user_query)
        logger.info(f"✓ User intent analyzed: {intent_analysis.get('main_focus', 'general')}")
        
        # Step 2: Collect results from all relevant agents
        agent_results = await self._collect_agent_results(session_id, agent_list)
        
        # Step 3: Basic confidence scoring (lightweight validation)
        validated_results = []
        for result in agent_results:
            # Simple confidence scoring based on data completeness
            confidence = self._calculate_basic_confidence(result)
            result.confidence_score = confidence
            validated_results.append(result)
            logger.info(f"✓ Processed results from {result.agent_name} (confidence: {confidence:.2f})")
        
        # Step 4: Generate LLM-driven report
        report_content = await self.report_generator.generate_report(
            user_query, 
            intent_analysis, 
            validated_results
        )
        
        # Step 5: Optional LLM validation (fact-checking)
        if os.getenv("ENABLE_FACT_CHECK", "true").lower() == "true":
            logger.info("🔍 Performing LLM-based fact checking...")
            validation_result = await self.validator.validate_with_llm(report_content, validated_results)
            
            if validation_result["validation_status"] == "needs_correction":
                logger.warning("⚠️ Report needed corrections after fact-checking")
                report_content = validation_result.get("validated_report", report_content)
            elif validation_result["validation_status"] == "valid":
                logger.info("✅ Report passed fact-checking validation")
            
            # Store validation info
            validation_info = {
                "validation_status": validation_result["validation_status"],
                "confidence_score": validation_result["confidence_score"],
                "issues_found": validation_result["issues_found"],
                "validation_notes": validation_result["validation_notes"]
            }
        else:
            validation_info = {"validation_status": "skipped", "confidence_score": 0.8}
        
        # Step 6: Aggregate visualizations and artifacts
        aggregated_visualizations = self.visualizer.aggregate(validated_results)
        dashboard_layout = self.visualizer.create_dashboard_layout(aggregated_visualizations)
        
        # Step 7: Compile final report package
        report_package = {
            "report_content": report_content,
            "metadata": {
                "session_id": session_id,
                "user_query": user_query,
                "timestamp": datetime.now().isoformat(),
                "generator": "CherryAI LLM-First Report Generator v1.0",
                "intent_analysis": intent_analysis,
                "validation_info": validation_info,
                "agent_count": len(validated_results)
            },
            "visualizations": aggregated_visualizations,
            "dashboard_layout": dashboard_layout,
            "artifacts": self._collect_all_artifacts(validated_results),
            "agent_performance": self._calculate_agent_metrics(validated_results)
        }
        
        logger.info(f"✓ LLM-driven report generated successfully with {len(validated_results)} agent results")
        
        return report_package
    
    async def _collect_agent_results(self, session_id: str, agent_list: Optional[List[str]] = None) -> List[AnalysisResult]:
        """Collect analysis results from multiple agents"""
        results = []
        agents_to_query = agent_list or list(AGENT_PORTS.keys())
        
        # Try to get results from session artifacts first
        session_data = await self.session_manager.get_session_data(session_id)
        
        for agent_name in agents_to_query:
            try:
                # Check if agent has artifacts in session
                agent_artifacts = await self._get_agent_artifacts(session_id, agent_name)
                
                if agent_artifacts:
                    result = AnalysisResult(
                        agent_name=agent_name,
                        timestamp=datetime.now().isoformat(),
                        status="completed",
                        data=agent_artifacts.get("data", {}),
                        artifacts=agent_artifacts.get("artifacts", []),
                        source_references=[f"session:{session_id}", f"agent:{agent_name}"]
                    )
                    results.append(result)
                    logger.info(f"✓ Collected results from {agent_name}")
                else:
                    logger.warning(f"✗ No artifacts found for {agent_name}")
                    
            except Exception as e:
                logger.error(f"Error collecting from {agent_name}: {e}")
        
        return results
    
    async def _get_agent_artifacts(self, session_id: str, agent_name: str) -> Optional[Dict]:
        """Retrieve artifacts from a specific agent"""
        # First try session-based artifacts
        artifacts_path = f"ai_ds_team/data/session_{session_id}/{agent_name}_artifacts.json"
        
        if os.path.exists(artifacts_path):
            with open(artifacts_path, "r") as f:
                return json.load(f)
        
        # Try shared dataframes directory
        shared_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/session_{session_id}_{agent_name}.json"
        
        if os.path.exists(shared_path):
            with open(shared_path, "r") as f:
                return json.load(f)
        
        return None
    
    def _calculate_basic_confidence(self, result: AnalysisResult) -> float:
        """Calculate basic confidence score based on data completeness"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on data completeness
        if result.data:
            confidence += 0.2
        
        if result.artifacts:
            confidence += 0.2
        
        if result.source_references:
            confidence += 0.1
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _calculate_agent_metrics(self, results: List[AnalysisResult]) -> List[Dict]:
        """Calculate performance metrics for each agent"""
        metrics = []
        
        for result in results:
            execution_time = result.data.get("execution_time", 0)
            if not execution_time and "timestamp" in result.data:
                # Estimate execution time
                execution_time = 1000  # Default 1 second
            
            metrics.append({
                "agent": result.agent_name,
                "execution_time": execution_time,
                "confidence": result.confidence_score
            })
        
        return metrics
    
    async def _assess_data_quality(self, session_id: str) -> Dict[str, Any]:
        """Assess overall data quality for the session"""
        quality_report = {
            "total_records": 0,
            "missing_values": 0,
            "outliers": 0
        }
        
        # Try to get data quality metrics from session data
        session_data = await self.session_manager.get_session_data(session_id)
        
        if session_data and "data_quality" in session_data:
            quality_report.update(session_data["data_quality"])
        else:
            # Default values if no quality data available
            quality_report = {
                "total_records": "N/A",
                "missing_values": "N/A",
                "outliers": "N/A"
            }
        
        return quality_report
    
    def _collect_all_artifacts(self, results: List[AnalysisResult]) -> List[Dict]:
        """Collect all artifacts from results"""
        all_artifacts = []
        
        for result in results:
            for artifact in result.artifacts:
                artifact["source_agent"] = result.agent_name
                all_artifacts.append(artifact)
        
        return all_artifacts
    
    async def invoke(self, query: str, session_id: Optional[str] = None) -> str:
        """Main entry point for LLM-driven report generation"""
        if not session_id:
            # Extract session ID from query if possible
            import re
            match = re.search(r'session[_-]?([a-f0-9\-]{8,})', query, re.IGNORECASE)
            if match:
                session_id = match.group(1)
            else:
                return "⚠️ 세션 ID가 필요합니다. 쿼리에 세션 ID를 포함하거나 별도로 제공해주세요."
        
        try:
            # Generate the LLM-driven report
            report_package = await self.generate_report(session_id, query)
            
            # Return the main report content
            return report_package["report_content"]
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"


class ReportGeneratorExecutor(AgentExecutor):
    """A2A Executor for Report Generator Agent"""
    
    def __init__(self):
        self.agent = ReportGeneratorAgent()
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute report generation request"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user query
            user_query = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_query += part.root.text + " "
            
            user_query = user_query.strip()
            if not user_query:
                await task_updater.reject(message="Query is empty")
                return
            
            logger.info(f"🚀 Report Generator processing: {user_query}")
            
            # Extract session ID from context or query
            session_id = context.metadata.get("session_id")
            if not session_id:
                # Try to extract from query
                import re
                match = re.search(r'session[_-]?([a-f0-9\-]+)', user_query, re.IGNORECASE)
                if match:
                    session_id = match.group(1)
            
            if not session_id:
                await task_updater.reject(message="Session ID not found. Please provide a session ID.")
                return
            
            # Generate LLM-driven report
            report_package = await self.agent.generate_report(session_id, user_query)
            
            # Send report content as response
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(report_package["report_content"])
            )
            
            # Log additional information
            metadata = report_package.get("metadata", {})
            logger.info(f"Report generated with intent: {metadata.get('intent_analysis', {}).get('main_focus', 'unknown')}")
            logger.info(f"Validation status: {metadata.get('validation_info', {}).get('validation_status', 'unknown')}")
            
            # Optionally store artifacts
            if "artifacts" in report_package:
                logger.info(f"Generated {len(report_package['artifacts'])} artifacts")
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            await task_updater.reject(message=f"Report generation failed: {str(e)}")
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation"""
        logger.info(f"Cancelling task {context.task_id}")


# Initialize the A2A application
def create_app():
    """Create and configure the A2A application"""
    # Agent metadata
    agent_card = AgentCard(
        name="report_generator",
        description="Generates comprehensive, accurate data analysis reports by synthesizing results from multiple analysis agents",
        url="http://localhost:8315/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="generate_report",
                name="Generate Comprehensive Report",
                description="Generate comprehensive, evidence-based reports from multiple analysis results",
                tags=["report", "synthesis", "analysis", "documentation"],
                examples=["generate report", "create summary", "synthesize results"]
            ),
            AgentSkill(
                id="validate_results",
                name="Validate Analysis Results",
                description="LLM-based validation of analysis results to prevent hallucinations",
                tags=["validation", "verification", "accuracy", "quality"],
                examples=["validate results", "check accuracy", "verify findings"]
            ),
            AgentSkill(
                id="aggregate_visualizations",
                name="Aggregate Visualizations",
                description="Combine and organize visualizations from multiple agents",
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
    
    # Create task store
    task_store = InMemoryTaskStore()
    
    # Create executor
    executor = ReportGeneratorExecutor()
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    # Create A2A app
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    return app


# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("REPORT_GENERATOR_PORT", 8315))
    logger.info(f"🚀 Starting Report Generator Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)