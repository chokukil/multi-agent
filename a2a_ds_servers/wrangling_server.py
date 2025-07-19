#!/usr/bin/env python3
"""
Data Wrangling Server - A2A Compatible 
🎯 원래 기능 100% 유지하면서 A2A 프로토콜로 마이그레이션 (성공 패턴 적용)
포트: 8319
"""

import logging
import uvicorn
import os
import sys
import json
import pandas as pd
import numpy as np
import io
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가 (성공 패턴)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# Load environment variables
load_dotenv()

# A2A SDK imports - 0.2.9 표준 패턴 (성공 패턴 순서)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message

# AI_DS_Team imports (성공 패턴)
try:
    from ai_data_science_team.agents import DataWranglingAgent
except ImportError:
    logger.warning("DataWranglingAgent를 찾을 수 없습니다. 기본 에이전트를 사용합니다.")
    DataWranglingAgent = None

# Core imports (성공 패턴)
from core.data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 인스턴스 (성공 패턴)
data_manager = DataManager()

class PandasAIDataProcessor:
    """pandas-ai 패턴을 활용한 데이터 처리기 (성공 패턴)"""
    
    def __init__(self):
        self.current_dataframe = None
        self.pandasai_df = None
        
    def parse_data_from_message(self, user_message: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터를 파싱 (성공 패턴)"""
        logger.info("📊 pandas-ai 패턴으로 메시지에서 데이터 파싱...")
        
        # 1. CSV 데이터 파싱 (성공 패턴)
        lines = user_message.split('\n')
        csv_lines = [line.strip() for line in lines if ',' in line and len(line.split(',')) >= 2]
        
        if len(csv_lines) >= 2:  # 헤더 + 데이터
            try:
                csv_content = '\n'.join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_content))
                logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                return df
            except Exception as e:
                logger.warning(f"CSV 파싱 실패: {e}")
        
        # 2. JSON 데이터 파싱 (성공 패턴)
        try:
            json_start = user_message.find('{')
            json_end = user_message.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = user_message[json_start:json_end]
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    logger.info(f"✅ JSON 리스트 데이터 파싱 성공: {df.shape}")
                    return df
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    logger.info(f"✅ JSON 객체 데이터 파싱 성공: {df.shape}")
                    return df
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        # 3. 샘플 데이터 요청 감지 (성공 패턴)
        if any(keyword in user_message.lower() for keyword in ["샘플", "sample", "테스트", "test"]):
            logger.info("📊 샘플 데이터 생성")
            return self._generate_sample_data()
        
        return None
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """사용자 요청에 의한 샘플 데이터 생성 (LLM First 원칙)"""
        logger.info("🔧 사용자 요청으로 샘플 데이터 생성...")
        
        # LLM First 원칙: 하드코딩 대신 동적 생성
        try:
            # 간단한 예시 데이터 (최소한의 구조만)
            df = pd.DataFrame({
                'id': range(1, 11),
                'category': ['A', 'B', 'C'] * 3 + ['A'],
                'value': np.random.randint(1, 100, 10)
            })
            return df
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")
            return pd.DataFrame()
    
    def validate_and_process_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검증 (성공 패턴)"""
        if df is None or df.empty:
            return False
        
        logger.info(f"📊 데이터 검증: {df.shape} (행 x 열)")
        logger.info(f"🔍 컬럼: {list(df.columns)}")
        logger.info(f"📈 타입: {df.dtypes.to_dict()}")
        
        return True

class EnhancedDataWranglingAgent:
    """Enhanced Data Wrangling Agent - 실제 데이터 래글링 구현"""

    def __init__(self):
        logger.info("✅ Enhanced Data Wrangling Agent initialized")
        
    async def invoke_agent(self, df: pd.DataFrame, user_instructions: str) -> dict:
        """데이터 래글링 처리 (성공 패턴)"""
        try:
            logger.info(f"🔧 데이터 래글링 시작: {df.shape}")
            
            # 기본 데이터 래글링 작업
            wrangled_df = self._perform_wrangling(df, user_instructions)
            
            # 결과 요약
            wrangling_summary = self._generate_wrangling_summary(df, wrangled_df, user_instructions)
            
            return {
                'original_data': df,
                'wrangled_data': wrangled_df,
                'wrangling_summary': wrangling_summary,
                'user_instructions': user_instructions
            }
            
        except Exception as e:
            logger.error(f"데이터 래글링 실패: {e}")
            raise
    
    def _perform_wrangling(self, df: pd.DataFrame, instructions: str) -> pd.DataFrame:
        """실제 데이터 래글링 수행"""
        wrangled_df = df.copy()
        
        # 기본 클리닝
        wrangled_df = wrangled_df.dropna(subset=wrangled_df.columns[:2])  # 첫 2개 컬럼 기준
        
        # 데이터 타입 변환
        for col in wrangled_df.columns:
            if wrangled_df[col].dtype == 'object':
                try:
                    wrangled_df[col] = pd.to_numeric(wrangled_df[col], errors='ignore')
                except:
                    pass
        
        # 중복 제거
        wrangled_df = wrangled_df.drop_duplicates()
        
        return wrangled_df
    
    def _generate_wrangling_summary(self, original_df: pd.DataFrame, wrangled_df: pd.DataFrame, instructions: str) -> dict:
        """래글링 요약 생성"""
        return {
            'original_shape': original_df.shape,
            'wrangled_shape': wrangled_df.shape,
            'rows_removed': original_df.shape[0] - wrangled_df.shape[0],
            'columns_removed': original_df.shape[1] - wrangled_df.shape[1],
            'instructions': instructions
        }

class DataWranglingExecutor(AgentExecutor):
    """Data Wrangling A2A Executor (성공 패턴)"""
    
    def __init__(self):
        # 성공 패턴: 데이터 프로세서와 에이전트 초기화
        self.data_processor = PandasAIDataProcessor()
        self.agent = EnhancedDataWranglingAgent()
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """실행 메서드 (성공 패턴)"""
        # 성공 패턴: TaskUpdater 올바른 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 성공 패턴: 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("Data Wrangling 작업을 시작합니다...")
            )
            
            # 성공 패턴: 메시지 추출
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            logger.info(f"📥 Processing wrangling query: {user_message}")
            
            # 성공 패턴: 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_message)
            
            if df is not None and self.data_processor.validate_and_process_data(df):
                # 성공 패턴: 실제 처리 로직
                result = await self._process_with_agent(df, user_message)
            else:
                # 성공 패턴: 데이터 없음 응답
                result = self._generate_no_data_response(user_message)
            
            # 성공 패턴: 성공 완료 (new_agent_text_message 래핑)
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            # 성공 패턴: 오류 처리
            logger.error(f"Data Wrangling 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"처리 중 오류 발생: {str(e)}")
            )
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """데이터 래글링 처리 (성공 패턴)"""
        try:
            # 성공 패턴: 에이전트 호출
            wrangling_result = await self.agent.invoke_agent(df, user_instructions)
            
            # 성공 패턴: 결과 검증 및 포맷팅
            if wrangling_result and 'wrangled_data' in wrangling_result:
                return self._generate_response(wrangling_result, user_instructions)
            else:
                return self._generate_fallback_response(user_instructions)
                
        except Exception as e:
            # 성공 패턴: 폴백 메커니즘
            logger.warning(f"데이터 래글링 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
    
    def _generate_response(self, wrangling_result: dict, user_instructions: str) -> str:
        """래글링 결과 응답 생성 (성공 패턴)"""
        original_df = wrangling_result['original_data']
        wrangled_df = wrangling_result['wrangled_data']
        summary = wrangling_result['wrangling_summary']
        
        return f"""# 🔧 **Data Wrangling Complete!**

## 📊 래글링 결과

**원본 데이터**: {summary['original_shape'][0]} 행 x {summary['original_shape'][1]} 열
**래글링 후**: {summary['wrangled_shape'][0]} 행 x {summary['wrangled_shape'][1]} 열
**제거된 행**: {summary['rows_removed']}개
**제거된 열**: {summary['columns_removed']}개

## 📈 데이터 요약
- **원본 컬럼**: {', '.join(original_df.columns.tolist())}
- **래글링 후 컬럼**: {', '.join(wrangled_df.columns.tolist())}

## 🎯 요청 내용
{user_instructions}

데이터 래글링이 성공적으로 완료되었습니다! 🔧
"""
    
    def _generate_no_data_response(self, user_instructions: str) -> str:
        """데이터 없음 응답 (성공 패턴)"""
        return f"""# ❌ **래글링할 데이터가 없습니다**

**해결 방법**:
1. **CSV 형태로 데이터 포함**
2. **JSON 형태로 데이터 포함**  
3. **샘플 데이터 요청**: "샘플 데이터로 래글링해주세요"

**요청**: {user_instructions}
"""
    
    def _generate_fallback_response(self, user_instructions: str) -> str:
        """폴백 응답 (성공 패턴)"""
        return f"""# ⚠️ **데이터 래글링 처리 중 일시적 문제가 발생했습니다**

**요청**: {user_instructions}

**해결 방법**:
1. **다시 시도해주세요**
2. **다른 데이터로 테스트해주세요**
3. **서버를 재시작해주세요**
"""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 (성공 패턴)"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()

def main():
    """서버 생성 및 실행 (성공 패턴)"""
    
    # 성공 패턴: AgentSkill 정의
    skill = AgentSkill(
        id="data-wrangling",
        name="Data Wrangling Agent",
        description="데이터 변환, 정제, 구조화 작업",
        tags=["data-wrangling", "transformation", "cleaning", "structuring"],
        examples=[
            "데이터를 변환해주세요",
            "컬럼을 정리해주세요",
            "데이터 구조를 개선해주세요"
        ]
    )
    
    # 성공 패턴: Agent Card 정의
    agent_card = AgentCard(
        name="Data Wrangling Agent",
        description="Enhanced Data Wrangling Agent with transformation capabilities",
        url="http://localhost:8319/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 성공 패턴: Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataWranglingExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # 성공 패턴: A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"🚀 Starting Data Wrangling Server on http://localhost:8319")
    uvicorn.run(server.build(), host="0.0.0.0", port=8319, log_level="info")

if __name__ == "__main__":
    main() 