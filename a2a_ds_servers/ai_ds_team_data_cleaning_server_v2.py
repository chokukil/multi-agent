#!/usr/bin/env python3
"""
AI_DS_Team DataCleaningAgent A2A Server V2 (Improved with Base Wrapper)
Port: 8306

AI_DS_Team의 DataCleaningAgent를 새로운 베이스 래퍼를 통해 A2A 프로토콜로 제공합니다.
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
from ai_data_science_team.agents import DataCleaningAgent

# 새로운 베이스 래퍼 사용
from base import AIDataScienceTeamWrapper

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 설정
from dotenv import load_dotenv
load_dotenv()


class DataCleaningWrapper(AIDataScienceTeamWrapper):
    """Data Cleaning Agent를 위한 특화된 래퍼"""
    
    def __init__(self):
        # LLM 인스턴스 생성
        from core.llm_factory import create_llm_instance
        llm = create_llm_instance()
        
        # 에이전트 설정
        agent_config = {
            "model": llm,
            "log": True,
            "log_path": "logs/generated_code/"
        }
        
        super().__init__(
            agent_class=DataCleaningAgent,
            agent_config=agent_config,
            agent_name="Data Cleaning Agent"
        )
    
    async def _execute_agent(self, user_input: str) -> any:
        """Data Cleaning Agent 특화 실행 로직"""
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
                    "error": "데이터 정리를 수행하려면 먼저 데이터를 업로드해야 합니다.",
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
            
            # Data Cleaning Agent 실행
            result = self.agent.invoke_agent(
                user_instructions=user_input,
                data_raw=df
            )
            
            # 정리된 데이터 가져오기
            cleaned_data = None
            try:
                cleaned_data = self.agent.get_data_cleaned()
            except Exception as e:
                logger.warning(f"Could not get cleaned data: {e}")
            
            return {
                "result": result,
                "original_data": df,
                "cleaned_data": cleaned_data,
                "data_file": data_file,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Data cleaning execution failed: {e}")
            return {
                "error": str(e),
                "content": f"데이터 정리 중 오류가 발생했습니다: {str(e)}",
                "success": False
            }
    
    def _build_final_response(self, workflow_summary: str, a2a_response: dict, user_input: str) -> str:
        """Data Cleaning 결과에 특화된 응답 구성"""
        try:
            if not a2a_response.get("success"):
                return f"""## ❌ 데이터 정리 실패

{a2a_response.get('content', '알 수 없는 오류가 발생했습니다.')}

요청: {user_input}

### 🧹 Data Cleaning Agent 사용법
1. **기본 정리**: "데이터를 정리해주세요"
2. **결측값 처리**: "결측값을 처리해주세요"
3. **중복값 제거**: "중복값을 제거해주세요"
4. **이상값 처리**: "이상값을 탐지하고 처리해주세요"
"""
            
            # 성공적인 경우
            original_data = a2a_response.get("original_data")
            cleaned_data = a2a_response.get("cleaned_data")
            data_file = a2a_response.get("data_file", "unknown")
            
            response_parts = [
                "## 🧹 데이터 정리 완료\n",
                f"### 📋 작업 요약\n{workflow_summary}\n"
            ]
            
            # 원본 데이터 정보
            if original_data is not None:
                response_parts.append(f"### 📊 원본 데이터 정보")
                response_parts.append(f"- **파일**: {data_file}")
                response_parts.append(f"- **크기**: {original_data.shape[0]:,} rows × {original_data.shape[1]:,} columns")
                response_parts.append(f"- **컬럼**: {', '.join(original_data.columns.tolist()[:5])}{'...' if len(original_data.columns) > 5 else ''}")
                response_parts.append(f"- **결측값**: {original_data.isnull().sum().sum():,}개\n")
            
            # 정리된 데이터 정보
            if cleaned_data is not None:
                response_parts.append(f"### 🔧 정리된 데이터 정보")
                response_parts.append(f"- **크기**: {cleaned_data.shape[0]:,} rows × {cleaned_data.shape[1]:,} columns")
                response_parts.append(f"- **결측값**: {cleaned_data.isnull().sum().sum():,}개")
                
                # 변화 요약
                if original_data is not None:
                    row_change = cleaned_data.shape[0] - original_data.shape[0]
                    missing_change = original_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum()
                    response_parts.append(f"- **행 변화**: {row_change:+,}")
                    response_parts.append(f"- **결측값 감소**: {missing_change:+,}개\n")
            
            response_parts.append("### 🧹 Data Cleaning Agent 기능")
            response_parts.append("- **결측값 처리**: fillna, dropna, 보간법 등")
            response_parts.append("- **중복 제거**: drop_duplicates 최적화")
            response_parts.append("- **이상값 탐지**: IQR, Z-score, Isolation Forest")
            response_parts.append("- **데이터 타입 변환**: 메모리 효율적인 타입 선택")
            response_parts.append("- **텍스트 정리**: 공백 제거, 대소문자 통일")
            response_parts.append("- **날짜 형식 표준화**: datetime 변환 및 검증")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error building data cleaning response: {e}")
            return f"✅ 데이터 정리 작업이 완료되었습니다.\n\n요청: {user_input}"


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_cleaning",
        name="Data Cleaning & Quality Improvement",
        description="Advanced data cleaning and quality improvement using AI-powered techniques",
        tags=["data", "cleaning", "preprocessing", "quality"],
        examples=[
            "Clean the dataset and remove missing values",
            "Remove duplicates and handle outliers",
            "Standardize data types and formats"
        ]
    )
    
    # AgentCard 생성
    agent_card = AgentCard(
        name="AI Data Science Team - Data Cleaning Agent",
        description="Specialized agent for data cleaning and quality improvement using advanced AI techniques",
        url="http://localhost:8306/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 요청 핸들러 설정
    request_handler = DefaultRequestHandler(
        agent_executor=DataCleaningWrapper(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A 서버 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🧹 Starting Data Cleaning Agent Server V2")
    print("🌐 Server starting on http://localhost:8306")
    print("📋 Agent card: http://localhost:8306/.well-known/agent.json")
    print("🔧 Using improved base wrapper architecture")
    
    # 서버 실행
    uvicorn.run(server.build(), host="0.0.0.0", port=8306, log_level="info")


if __name__ == "__main__":
    main() 