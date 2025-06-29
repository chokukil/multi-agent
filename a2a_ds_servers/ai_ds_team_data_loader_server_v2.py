#!/usr/bin/env python3
"""
AI_DS_Team DataLoaderToolsAgent A2A Server V2 (Improved with Base Wrapper)
Port: 8307

AI_DS_Team의 DataLoaderToolsAgent를 새로운 베이스 래퍼를 통해 A2A 프로토콜로 제공합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))
sys.path.insert(0, str(project_root / "a2a_ds_servers"))

import uvicorn
import logging
import pandas as pd

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

# AI_DS_Team imports
from ai_data_science_team.agents import DataLoaderToolsAgent

# 새로운 베이스 래퍼 사용
from base import AIDataScienceTeamWrapper

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 설정
from dotenv import load_dotenv
load_dotenv()


class DataLoaderWrapper(AIDataScienceTeamWrapper):
    """Data Loader Agent를 위한 특화된 래퍼"""
    
    def __init__(self):
        # LLM 인스턴스 생성
        from core.llm_factory import create_llm_instance
        llm = create_llm_instance()
        
        # 에이전트 설정
        agent_config = {
            "model": llm
        }
        
        super().__init__(
            agent_class=DataLoaderToolsAgent,
            agent_config=agent_config,
            agent_name="Data Loader Tools Agent"
        )
    
    async def _execute_agent(self, user_input: str) -> any:
        """Data Loader Agent 특화 실행 로직"""
        try:
            # Data Loader Agent 실행
            result = self.agent.invoke_agent(
                user_instructions=user_input
            )
            
            # 로드된 데이터 확인
            loaded_data = None
            if hasattr(self.agent, 'data') and self.agent.data is not None:
                loaded_data = self.agent.data
                
                # 데이터를 공유 폴더에 저장
                data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
                os.makedirs(data_path, exist_ok=True)
                
                # 고유한 파일명 생성
                import time
                timestamp = int(time.time())
                output_file = f"loaded_data_{timestamp}.csv"
                output_path = os.path.join(data_path, output_file)
                
                loaded_data.to_csv(output_path, index=False)
                logger.info(f"Data saved to: {output_path}")
            
            # 사용 가능한 데이터 소스 검색
            available_sources = self._scan_available_data_sources()
            
            return {
                "result": result,
                "loaded_data": loaded_data,
                "data_file": output_file if loaded_data is not None else None,
                "available_sources": available_sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Data loading execution failed: {e}")
            return {
                "error": str(e),
                "content": f"데이터 로딩 중 오류가 발생했습니다: {str(e)}",
                "success": False
            }
    
    def _scan_available_data_sources(self) -> list:
        """사용 가능한 데이터 소스를 스캔합니다."""
        available_sources = []
        
        # 스캔할 디렉토리들
        data_dirs = [
            "ai_ds_team/data/",
            "a2a_ds_servers/artifacts/data/shared_dataframes/",
            "data/",
            "artifacts/data/shared_dataframes/"
        ]
        
        for data_dir in data_dirs:
            try:
                if os.path.exists(data_dir):
                    files = [f for f in os.listdir(data_dir) 
                            if f.endswith(('.csv', '.xlsx', '.json', '.parquet', '.pkl'))]
                    if files:
                        for file in files:
                            available_sources.append({
                                "path": os.path.join(data_dir, file),
                                "name": file,
                                "directory": data_dir,
                                "size": self._get_file_size(os.path.join(data_dir, file))
                            })
            except Exception as e:
                logger.warning(f"Error scanning directory {data_dir}: {e}")
        
        return available_sources
    
    def _get_file_size(self, file_path: str) -> str:
        """파일 크기를 사람이 읽기 쉬운 형태로 반환합니다."""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.1f} MB"
            else:
                return f"{size_bytes/(1024**3):.1f} GB"
        except:
            return "Unknown"
    
    def _build_final_response(self, workflow_summary: str, a2a_response: dict, user_input: str) -> str:
        """Data Loading 결과에 특화된 응답 구성"""
        try:
            if not a2a_response.get("success"):
                return f"""## ❌ 데이터 로딩 실패

{a2a_response.get('content', '알 수 없는 오류가 발생했습니다.')}

요청: {user_input}

### 💡 Data Loader Tools 사용법
1. **파일 로딩**: "CSV 파일을 로드해주세요"
2. **데이터 검색**: "사용 가능한 데이터 파일들을 보여주세요"
3. **형식 변환**: "JSON을 DataFrame으로 변환해주세요"
4. **데이터베이스**: "데이터베이스에서 테이블을 가져와주세요"
"""
            
            # 성공적인 경우
            loaded_data = a2a_response.get("loaded_data")
            data_file = a2a_response.get("data_file")
            available_sources = a2a_response.get("available_sources", [])
            
            response_parts = [
                "## 📁 데이터 로딩 완료\n",
                f"### 📋 작업 요약\n{workflow_summary}\n"
            ]
            
            # 로드된 데이터 정보
            if loaded_data is not None:
                response_parts.append(f"### 📊 로드된 데이터 정보")
                response_parts.append(f"- **저장된 파일**: `{data_file}`")
                response_parts.append(f"- **데이터 크기**: {loaded_data.shape[0]:,} rows × {loaded_data.shape[1]:,} columns")
                response_parts.append(f"- **메모리 사용량**: {loaded_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                response_parts.append(f"- **컬럼**: {', '.join(loaded_data.columns.tolist()[:5])}{'...' if len(loaded_data.columns) > 5 else ''}")
                
                # 데이터 타입 정보
                if len(loaded_data.columns) <= 10:
                    response_parts.append(f"- **데이터 타입**: {dict(loaded_data.dtypes.astype(str))}")
                else:
                    response_parts.append(f"- **데이터 타입**: {len(loaded_data.select_dtypes('number').columns)} 숫자형, {len(loaded_data.select_dtypes('object').columns)} 문자형")
                
                response_parts.append("")
            
            # 사용 가능한 데이터 소스 정보
            if available_sources:
                response_parts.append("### 📁 사용 가능한 데이터 소스")
                for source in available_sources[:8]:  # 최대 8개까지만 표시
                    response_parts.append(f"- **{source['name']}** ({source['size']}) - `{source['directory']}`")
                
                if len(available_sources) > 8:
                    response_parts.append(f"- ... 및 {len(available_sources) - 8}개 추가 파일")
                
                response_parts.append("")
            
            response_parts.append("### 🛠️ Data Loader Tools 기능")
            response_parts.append("- **파일 로딩**: CSV, Excel, JSON, Parquet 등 다양한 형식 지원")
            response_parts.append("- **데이터베이스 연결**: SQL 데이터베이스 연결 및 쿼리")
            response_parts.append("- **API 통합**: REST API를 통한 데이터 수집")
            response_parts.append("- **데이터 검증**: 로드된 데이터의 품질 및 형식 검증")
            response_parts.append("- **자동 타입 추론**: 컬럼 타입 자동 감지 및 변환")
            response_parts.append("- **배치 처리**: 대용량 파일의 청크 단위 로딩")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error building data loader response: {e}")
            return f"✅ 데이터 로딩 작업이 완료되었습니다.\n\n요청: {user_input}"


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_loading",
        name="Data Loading & File Processing",
        description="Advanced data loading and file processing with support for multiple formats and sources",
        tags=["data-loading", "etl", "file-processing", "database", "api-integration"],
        examples=[
            "Load CSV file and convert to DataFrame",
            "Connect to database and fetch customer table",
            "Collect real-time data from API endpoints",
            "Read specific sheet from Excel file",
            "Show available data files in the system"
        ]
    )
    
    # AgentCard 생성
    agent_card = AgentCard(
        name="AI Data Science Team - Data Loader Tools Agent",
        description="Specialized agent for data loading and file processing with advanced ETL capabilities",
        url="http://localhost:8307/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 요청 핸들러 설정
    request_handler = DefaultRequestHandler(
        agent_executor=DataLoaderWrapper(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A 서버 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("📁 Starting Data Loader Tools Agent Server V2")
    print("🌐 Server starting on http://localhost:8307")
    print("📋 Agent card: http://localhost:8307/.well-known/agent.json")
    print("🔧 Using improved base wrapper architecture")
    
    # 서버 실행
    uvicorn.run(server.build(), host="0.0.0.0", port=8307, log_level="info")


if __name__ == "__main__":
    main() 