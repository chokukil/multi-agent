#!/usr/bin/env python3
"""
AI_DS_Team EDAToolsAgent A2A Server V2 (Improved with Base Wrapper)
Port: 8312

AI_DS_Team의 EDAToolsAgent를 새로운 베이스 래퍼를 통해 A2A 프로토콜로 제공합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

import uvicorn
import logging
import pandas as pd

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

# AI_DS_Team imports
from ai_data_science_team.ds_agents import EDAToolsAgent

# 새로운 베이스 래퍼 사용
from base import AIDataScienceTeamWrapper

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 설정
from dotenv import load_dotenv
load_dotenv()


class EDAToolsWrapper(AIDataScienceTeamWrapper):
    """EDA Tools Agent를 위한 특화된 래퍼"""
    
    def __init__(self):
        # LLM 인스턴스 생성
        from core.llm_factory import create_llm_instance
        llm = create_llm_instance()
        
        # 에이전트 설정
        agent_config = {
            "model": llm
        }
        
        super().__init__(
            agent_class=EDAToolsAgent,
            agent_config=agent_config,
            agent_name="EDA Tools Agent"
        )
    
    async def _execute_agent(self, user_input: str) -> any:
        """EDA Tools Agent 특화 실행 로직"""
        try:
            # 데이터 로드
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            available_data = []
            
            try:
                for file in os.listdir(data_path):
                    if file.endswith(('.csv', '.pkl')):
                        available_data.append(file)
            except:
                pass
            
            if not available_data:
                return {
                    "error": "탐색적 데이터 분석을 수행하려면 먼저 데이터를 업로드해야 합니다.",
                    "content": "사용 가능한 데이터가 없습니다.",
                    "success": False
                }
            
            # 가장 최근 데이터 사용
            data_file = available_data[0]
            if data_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_path, data_file))
            else:
                df = pd.read_pickle(os.path.join(data_path, data_file))
            
            logger.info(f"Loaded data: {data_file}, shape: {df.shape}")
            
            # EDA Tools Agent 실행
            result = self.agent.invoke_agent(
                user_instructions=user_input,
                data_raw=df
            )
            
            # 생성된 EDA 아티팩트 확인
            artifacts_info = self._collect_eda_artifacts()
            
            return {
                "result": result,
                "data": df,
                "data_file": data_file,
                "artifacts": artifacts_info,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"EDA execution failed: {e}")
            return {
                "error": str(e),
                "content": f"탐색적 데이터 분석 중 오류가 발생했습니다: {str(e)}",
                "success": False
            }
    
    def _collect_eda_artifacts(self) -> dict:
        """생성된 EDA 아티팩트 정보를 수집합니다."""
        artifacts_info = {
            "eda_reports": [],
            "plots": [],
            "data_files": []
        }
        
        try:
            # EDA 보고서 확인
            eda_path = "a2a_ds_servers/artifacts/eda/"
            if os.path.exists(eda_path):
                for file in os.listdir(eda_path):
                    if file.endswith('.html'):
                        artifacts_info["eda_reports"].append(file)
            
            # 플롯 파일 확인
            plots_path = "a2a_ds_servers/artifacts/plots/"
            if os.path.exists(plots_path):
                for file in os.listdir(plots_path):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                        artifacts_info["plots"].append(file)
            
            # 데이터 파일 확인
            data_path = "a2a_ds_servers/artifacts/data/"
            if os.path.exists(data_path):
                for file in os.listdir(data_path):
                    if file.endswith(('.csv', '.json', '.pkl')):
                        artifacts_info["data_files"].append(file)
                        
        except Exception as e:
            logger.warning(f"Error collecting artifacts: {e}")
        
        return artifacts_info
    
    def _build_final_response(self, workflow_summary: str, a2a_response: dict, user_input: str) -> str:
        """EDA 결과에 특화된 응답 구성"""
        try:
            if not a2a_response.get("success"):
                return f"""## ❌ 탐색적 데이터 분석 실패

{a2a_response.get('content', '알 수 없는 오류가 발생했습니다.')}

요청: {user_input}

### 🔍 EDA Tools Agent 사용법
1. **기본 EDA**: "데이터의 기본 통계와 분포를 분석해주세요"
2. **상관관계 분석**: "변수들 간의 상관관계를 분석해주세요"
3. **자동 보고서**: "Sweetviz 보고서를 생성해주세요"
4. **결측값 분석**: "결측값과 이상값을 확인해주세요"
"""
            
            # 성공적인 경우
            data = a2a_response.get("data")
            data_file = a2a_response.get("data_file", "unknown")
            artifacts = a2a_response.get("artifacts", {})
            
            response_parts = [
                "## 🔍 탐색적 데이터 분석(EDA) 완료\n",
                f"### 📋 분석 요약\n{workflow_summary}\n"
            ]
            
            # 데이터 정보
            if data is not None:
                response_parts.append(f"### 📊 분석된 데이터 정보")
                response_parts.append(f"- **파일**: {data_file}")
                response_parts.append(f"- **크기**: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
                response_parts.append(f"- **컬럼**: {', '.join(data.columns.tolist()[:5])}{'...' if len(data.columns) > 5 else ''}")
                response_parts.append(f"- **결측값**: {data.isnull().sum().sum():,}개")
                response_parts.append(f"- **중복값**: {data.duplicated().sum():,}개\n")
            
            # 생성된 아티팩트 정보
            if artifacts:
                response_parts.append("### 💾 생성된 분석 결과")
                
                if artifacts.get("eda_reports"):
                    response_parts.append("**📋 EDA 보고서:**")
                    for report in artifacts["eda_reports"][-3:]:  # 최근 3개만
                        response_parts.append(f"- {report}")
                
                if artifacts.get("plots"):
                    response_parts.append("**📈 생성된 차트:**")
                    for plot in artifacts["plots"][-5:]:  # 최근 5개만
                        response_parts.append(f"- {plot}")
                
                if artifacts.get("data_files"):
                    response_parts.append("**💾 데이터 파일:**")
                    for data_file in artifacts["data_files"][-3:]:  # 최근 3개만
                        response_parts.append(f"- {data_file}")
                
                response_parts.append("")
            
            response_parts.append("### 🧰 EDA Tools Agent 기능")
            response_parts.append("- **데이터 프로파일링**: 자동 데이터 품질 분석")
            response_parts.append("- **분포 분석**: 변수별 분포 및 통계 분석")
            response_parts.append("- **상관관계 분석**: Correlation Funnel 및 히트맵")
            response_parts.append("- **결측값 분석**: Missingno 시각화")
            response_parts.append("- **자동 보고서**: Sweetviz, Pandas Profiling")
            response_parts.append("- **통계적 검정**: 가설 검정 및 통계 분석")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error building EDA response: {e}")
            return f"✅ 탐색적 데이터 분석이 완료되었습니다.\n\n요청: {user_input}"


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="eda_tools",
        name="Exploratory Data Analysis Tools",
        description="Comprehensive EDA tools for data exploration and statistical analysis",
        tags=["eda", "analysis", "statistics", "visualization"],
        examples=[
            "Analyze data distribution and basic statistics",
            "Generate correlation analysis and heatmaps",
            "Create comprehensive EDA report with Sweetviz",
            "Analyze missing values and outliers"
        ]
    )
    
    # AgentCard 생성
    agent_card = AgentCard(
        name="AI Data Science Team - EDA Tools Agent",
        description="Specialized agent for exploratory data analysis with advanced statistical tools",
        url="http://localhost:8312/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 요청 핸들러 설정
    request_handler = DefaultRequestHandler(
        agent_executor=EDAToolsWrapper(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A 서버 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔍 Starting EDA Tools Agent Server V2")
    print("🌐 Server starting on http://localhost:8312")
    print("📋 Agent card: http://localhost:8312/.well-known/agent.json")
    print("🔧 Using improved base wrapper architecture")
    
    # 서버 실행
    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")


if __name__ == "__main__":
    main() 