#!/usr/bin/env python3
"""
Shared Knowledge Bank Server - A2A Compatible 
🎯 지식 저장 및 검색 기능 구현 (성공 패턴 적용)
포트: 8325
"""

import logging
import uvicorn
import os
import sys
import json
import uuid
import pandas as pd
import numpy as np
import io
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가 (성공 패턴)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# AI_DS_Team imports (성공 패턴)
try:
    # KnowledgeBankAgent는 존재하지 않으므로 기본 에이전트 사용
    KnowledgeBankAgent = None
except ImportError:
    KnowledgeBankAgent = None

# A2A imports (성공 패턴 순서)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message

# Core imports (성공 패턴)
from core.data_manager import DataManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

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
                'title': [f'Knowledge {i}' for i in range(1, 6)],
                'content': [f'This is knowledge content {i}' for i in range(1, 6)],
                'category': ['A', 'B', 'C', 'A', 'B']
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

class EnhancedKnowledgeBankAgent:
    """Enhanced Knowledge Bank Agent - 실제 지식 저장 및 검색 구현"""

    def __init__(self):
        self.knowledge_base = {}
        self.knowledge_id_counter = 1
        logger.info("✅ Enhanced Knowledge Bank Agent initialized")
        
    async def store_knowledge(self, df: pd.DataFrame, user_instructions: str) -> dict:
        """지식 저장 처리 (성공 패턴)"""
        try:
            logger.info(f"🧠 지식 저장 시작: {df.shape}")
            
            # 지식 항목 생성
            knowledge_entries = []
            for idx, row in df.iterrows():
                knowledge_id = f"kb_{self.knowledge_id_counter}"
                self.knowledge_id_counter += 1
                
                knowledge_entry = {
                    'id': knowledge_id,
                    'title': row.get('title', f'Knowledge {idx+1}'),
                    'content': row.get('content', str(row.to_dict())),
                    'category': row.get('category', 'general'),
                    'importance': row.get('importance', 5),
                    'created_at': datetime.now().isoformat(),
                    'metadata': row.to_dict()
                }
                
                self.knowledge_base[knowledge_id] = knowledge_entry
                knowledge_entries.append(knowledge_entry)
            
            return {
                'stored_entries': knowledge_entries,
                'total_knowledge': len(self.knowledge_base),
                'user_instructions': user_instructions
            }
            
        except Exception as e:
            logger.error(f"지식 저장 실패: {e}")
            raise
    
    async def search_knowledge(self, df: pd.DataFrame, user_instructions: str) -> dict:
        """지식 검색 처리 (성공 패턴)"""
        try:
            logger.info(f"🔍 지식 검색 시작: {df.shape}")
            
            # 검색 쿼리 추출
            search_query = user_instructions.lower()
            
            # 지식 검색
            search_results = []
            for knowledge_id, entry in self.knowledge_base.items():
                # 간단한 키워드 매칭
                if any(keyword in entry['title'].lower() or keyword in entry['content'].lower() 
                      for keyword in search_query.split()):
                    search_results.append(entry)
            
            return {
                'search_results': search_results,
                'total_found': len(search_results),
                'search_query': search_query,
                'user_instructions': user_instructions
            }
            
        except Exception as e:
            logger.error(f"지식 검색 실패: {e}")
            raise
    
class KnowledgeBankExecutor(AgentExecutor):
    """Knowledge Bank A2A Executor (성공 패턴)"""
    
    def __init__(self):
        # 성공 패턴: 데이터 프로세서와 에이전트 초기화
        self.data_processor = PandasAIDataProcessor()
        self.agent = EnhancedKnowledgeBankAgent()
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """실행 메서드 (성공 패턴)"""
        # 성공 패턴: TaskUpdater 올바른 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 성공 패턴: 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("Knowledge Bank 작업을 시작합니다...")
            )
            
            # 성공 패턴: 메시지 추출
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            logger.info(f"📥 Processing knowledge bank query: {user_message}")
            
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
            logger.error(f"Knowledge Bank 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"처리 중 오류 발생: {str(e)}")
            )
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """지식 저장/검색 처리 (성공 패턴)"""
        try:
            # 성공 패턴: 에이전트 호출
            if "검색" in user_instructions or "search" in user_instructions.lower():
                knowledge_result = await self.agent.search_knowledge(df, user_instructions)
            else:
                knowledge_result = await self.agent.store_knowledge(df, user_instructions)
            
            # 성공 패턴: 결과 검증 및 포맷팅
            if knowledge_result:
                return self._generate_response(knowledge_result, user_instructions)
            else:
                return self._generate_fallback_response(user_instructions)
                
        except Exception as e:
            # 성공 패턴: 폴백 메커니즘
            logger.warning(f"지식 저장/검색 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
    
    def _generate_response(self, knowledge_result: dict, user_instructions: str) -> str:
        """지식 결과 응답 생성 (성공 패턴)"""
        if 'stored_entries' in knowledge_result:
            # 저장 결과
            stored_entries = knowledge_result['stored_entries']
            total_knowledge = knowledge_result['total_knowledge']
            
            return f"""# 🧠 **Knowledge Bank Complete!**

## 📚 지식 저장 결과

**저장된 항목**: {len(stored_entries)}개
**총 지식 항목**: {total_knowledge}개

## 📝 저장된 지식
{chr(10).join([f"- **{entry['title']}**: {entry['content'][:100]}..." for entry in stored_entries[:5]])}

## 🎯 요청 내용
{user_instructions}

지식이 성공적으로 저장되었습니다! 🧠
"""
        else:
            # 검색 결과
            search_results = knowledge_result['search_results']
            total_found = knowledge_result['total_found']
            search_query = knowledge_result['search_query']
            
            return f"""# 🔍 **Knowledge Search Complete!**

## 📚 검색 결과

**검색어**: {search_query}
**찾은 항목**: {total_found}개

## 📝 검색된 지식
{chr(10).join([f"- **{entry['title']}**: {entry['content'][:100]}..." for entry in search_results[:5]])}

## 🎯 요청 내용
{user_instructions}

지식 검색이 완료되었습니다! 🔍
"""
    
    def _generate_no_data_response(self, user_instructions: str) -> str:
        """데이터 없음 응답 (성공 패턴)"""
        return f"""# ❌ **지식 저장/검색할 데이터가 없습니다**

**해결 방법**:
1. **CSV 형태로 데이터 포함**
2. **JSON 형태로 데이터 포함**  
3. **샘플 데이터 요청**: "샘플 데이터로 지식을 저장해주세요"

**요청**: {user_instructions}
"""
    
    def _generate_fallback_response(self, user_instructions: str) -> str:
        """폴백 응답 (성공 패턴)"""
        return f"""# ⚠️ **지식 저장/검색 처리 중 일시적 문제가 발생했습니다**

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
        id="knowledge-bank",
        name="Knowledge Bank Agent",
        description="지식 저장, 검색, 관리 기능",
        tags=["knowledge", "storage", "search", "management"],
        examples=[
            "지식을 저장해주세요",
            "지식을 검색해주세요",
            "샘플 데이터로 지식을 저장해주세요"
        ]
    )
    
    # 성공 패턴: Agent Card 정의
    agent_card = AgentCard(
        name="Knowledge Bank Agent",
        description="Enhanced Knowledge Bank Agent with storage and search capabilities",
        url="http://localhost:8325/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 성공 패턴: Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=KnowledgeBankExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # 성공 패턴: A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"🚀 Starting Knowledge Bank Server on http://localhost:8325")
    uvicorn.run(server.build(), host="0.0.0.0", port=8325, log_level="info")

if __name__ == "__main__":
    main() 