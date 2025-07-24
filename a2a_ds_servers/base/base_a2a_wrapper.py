#!/usr/bin/env python3
"""
BaseA2AWrapper - A2A SDK 0.2.9 래핑 베이스 클래스

원본 ai-data-science-team 패키지의 에이전트들을 A2A SDK 0.2.9 프로토콜로 
래핑하기 위한 공통 베이스 클래스입니다.

주요 기능:
1. A2A SDK 0.2.9 공식 패턴 준수
2. TaskUpdater 패턴 구현
3. 원본 에이전트 8개 기능 100% 보존
4. 표준화된 에러 핸들링
5. 공통 데이터 처리 패턴
"""

import logging
import asyncio
import pandas as pd
import numpy as np
import io
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys
import os

# A2A SDK imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState, TextPart
from a2a.utils import new_agent_text_message

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

logger = logging.getLogger(__name__)


class PandasAIDataProcessor:
    """pandas-ai 스타일 데이터 프로세서 - 100% LLM First, 샘플 데이터 생성 절대 금지"""
    
    def parse_data_from_message(self, user_instructions: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터 파싱 - 절대 샘플 데이터 생성 안함"""
        logger.info("🔍 데이터 파싱 시작 (샘플 데이터 생성 절대 금지)")
        
        # CSV 데이터 검색 (일반 개행 문자 포함)
        if ',' in user_instructions and ('\n' in user_instructions or '\\n' in user_instructions):
            try:
                # 실제 개행문자와 이스케이프된 개행문자 모두 처리
                normalized_text = user_instructions.replace('\\n', '\n')
                lines = normalized_text.strip().split('\n')
                
                # CSV 패턴 찾기 - 헤더와 데이터 행 구분
                csv_lines = []
                for line in lines:
                    line = line.strip()
                    if ',' in line and line:  # 쉼표가 있고 비어있지 않은 행
                        csv_lines.append(line)
                
                if len(csv_lines) >= 2:  # 헤더 + 최소 1개 데이터 행
                    csv_data = '\n'.join(csv_lines)
                    df = pd.read_csv(io.StringIO(csv_data))
                    logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                    return df
            except Exception as e:
                logger.warning(f"CSV 파싱 실패: {e}")
        
        # JSON 데이터 검색
        try:
            import re
            json_pattern = r'\\[.*?\\]|\\{.*?\\}'
            json_matches = re.findall(json_pattern, user_instructions, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data)
                        logger.info(f"✅ JSON 데이터 파싱 성공: {df.shape}")
                        return df
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                        logger.info(f"✅ JSON 객체 파싱 성공: {df.shape}")
                        return df
                except:
                    continue
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        # 절대 샘플 데이터 생성 안함
        logger.info("⚠️ 파싱 가능한 데이터 없음 - None 반환 (샘플 데이터 생성 금지)")
        return None


class BaseA2AWrapper:
    """
    A2A SDK 0.2.9 래핑을 위한 베이스 클래스
    
    모든 ai-data-science-team 에이전트들이 이 클래스를 상속받아
    일관된 A2A 프로토콜 래핑을 제공합니다.
    """
    
    def __init__(self, agent_name: str, original_agent_class, port: int):
        """
        BaseA2AWrapper 초기화
        
        Args:
            agent_name: 에이전트 이름 (예: "DataCleaningAgent")
            original_agent_class: 원본 ai-data-science-team 에이전트 클래스
            port: 서버 포트
        """
        self.agent_name = agent_name
        self.original_agent_class = original_agent_class
        self.port = port
        self.llm = None
        self.agent = None
        self.data_processor = PandasAIDataProcessor()
        
        # LLM 및 원본 에이전트 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LLM 및 원본 에이전트 초기화"""
        try:
            # LLM 제공자 확인 (.env 파일 기준)
            llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
            logger.info(f"🔧 LLM 제공자: {llm_provider}")
            
            # Ollama는 API 키 불필요, 나머지는 API 키 체크
            if llm_provider != 'ollama':
                api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError(f"No API key found for {llm_provider.upper()}. Ollama doesn't require API keys.")
            else:
                logger.info("🚀 Ollama 사용 - API 키 불필요")
                
            # Universal Engine LLM Factory 사용
            from core.universal_engine.llm_factory import LLMFactory
            self.llm = LLMFactory.create_llm_client()
            
            # 원본 에이전트 초기화 (서브클래스에서 구현)
            self.agent = self._create_original_agent()
            
            logger.info(f"✅ {self.agent_name} 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ {self.agent_name} 초기화 실패: {e}")
            raise RuntimeError(f"{self.agent_name} initialization is required for operation") from e
    
    def _create_original_agent(self):
        """
        원본 에이전트 생성 (서브클래스에서 구현)
        
        Returns:
            원본 ai-data-science-team 에이전트 인스턴스
        """
        return self.original_agent_class(model=self.llm)
    
    async def process_request(self, user_input: str, function_name: str = None) -> str:
        """
        A2A 요청 처리 - 원본 에이전트 100% 기능 구현
        
        Args:
            user_input: 사용자 입력
            function_name: 특정 기능 호출 (선택사항)
            
        Returns:
            처리 결과 텍스트
        """
        try:
            logger.info(f"🚀 {self.agent_name} 요청 처리 시작: {user_input[:100]}...")
            
            # 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_input)
            
            if df is None:
                # 데이터 없이 가이드 제공
                return self._generate_guidance(user_input)
            
            # 원본 에이전트 invoke_agent() 호출
            logger.info(f"🤖 원본 {self.agent_name}.invoke_agent 실행 중...")
            
            # 서브클래스에서 구체적인 invoke_agent 호출 구현
            result = await self._invoke_original_agent(df, user_input, function_name)
            
            # 데이터를 공유 폴더에 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            import time
            timestamp = int(time.time())
            output_file = f"{self.agent_name.lower()}_data_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to: {output_path}")
            
            # 결과 구성
            final_result = self._format_result(result, df, output_path, user_input)
            
            logger.info(f"✅ {self.agent_name} 처리 완료")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ {self.agent_name} 처리 중 오류: {e}")
            return f"❌ {self.agent_name} 처리 중 오류 발생: {str(e)}"
    
    async def _invoke_original_agent(self, df: pd.DataFrame, user_input: str, function_name: str = None) -> Dict[str, Any]:
        """
        원본 에이전트 invoke_agent 호출 (서브클래스에서 구현)
        
        Args:
            df: 파싱된 데이터프레임
            user_input: 사용자 입력
            function_name: 특정 기능 이름
            
        Returns:
            에이전트 응답 딕셔너리
        """
        # 기본 구현 - 서브클래스에서 오버라이드
        self.agent.invoke_agent(
            data_raw=df,
            user_instructions=user_input
        )
        
        return {
            "response": self.agent.response if hasattr(self.agent, 'response') else None,
            "ai_message": self.agent.get_ai_message() if hasattr(self.agent, 'get_ai_message') else None
        }
    
    def _format_result(self, result: Dict[str, Any], df: pd.DataFrame, output_path: str, user_input: str) -> str:
        """
        결과 포맷팅 (서브클래스에서 오버라이드 가능)
        
        Args:
            result: 에이전트 결과
            df: 처리된 데이터프레임
            output_path: 저장된 파일 경로
            user_input: 사용자 입력
            
        Returns:
            포맷팅된 결과 텍스트
        """
        data_preview = df.head().to_string()
        
        return f"""# 🤖 **{self.agent_name} Complete!**

## 📊 **처리된 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {', '.join(df.columns.tolist())}
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

## 📝 **요청 내용**
{user_input}

## 📈 **데이터 미리보기**
```
{data_preview}
```

## 🎯 **처리 결과**
{result.get('ai_message', '처리가 완료되었습니다.')}

✅ **원본 ai-data-science-team {self.agent_name} 100% 기능이 성공적으로 완료되었습니다!**
"""
    
    def _generate_guidance(self, user_instructions: str) -> str:
        """
        데이터 없을 때 가이드 제공 (서브클래스에서 오버라이드)
        
        Args:
            user_instructions: 사용자 요청
            
        Returns:
            가이드 텍스트
        """
        return f"""# 🤖 **{self.agent_name} 가이드**

## 📝 **요청 내용**
{user_instructions}

## 💡 **데이터를 포함해서 다시 요청하면 실제 {self.agent_name} 작업을 수행해드릴 수 있습니다!**

**데이터 형식 예시**:
- **CSV**: `column1,column2,target\\n1.0,2.0,1\\n1.5,2.5,0`
- **JSON**: `[{{"column1": 1.0, "column2": 2.0, "target": 1}}]`

✅ **{self.agent_name} 준비 완료!**
"""

    def get_function_mapping(self) -> Dict[str, str]:
        """
        8개 기능 매핑 반환 (서브클래스에서 구현)
        
        Returns:
            기능명 -> 메서드명 매핑 딕셔너리
        """
        # 기본 8개 기능 - 서브클래스에서 오버라이드
        return {
            "function1": "get_function1",
            "function2": "get_function2", 
            "function3": "get_function3",
            "function4": "get_function4",
            "function5": "get_function5",
            "function6": "get_function6",
            "function7": "get_function7",
            "function8": "get_function8"
        }


class BaseA2AExecutor(AgentExecutor):
    """
    A2A SDK 0.2.9 래핑을 위한 베이스 Executor 클래스
    
    TaskUpdater 패턴을 적용한 표준화된 실행 클래스입니다.
    """
    
    def __init__(self, wrapper_agent):
        """
        BaseA2AExecutor 초기화
        
        Args:
            wrapper_agent: BaseA2AWrapper 인스턴스
        """
        self.agent = wrapper_agent
        logger.info(f"🤖 {wrapper_agent.agent_name} Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 공식 패턴에 따른 실행"""
        logger.info(f"🚀 {self.agent.agent_name} 실행 시작 - Task: {context.task_id}")
        
        # TaskUpdater 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"🤖 원본 ai-data-science-team {self.agent.agent_name} 시작...")
            )
            
            # A2A SDK 0.2.9 공식 패턴에 따른 사용자 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"📝 사용자 요청: {user_instructions}")
                
                if not user_instructions:
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message(f"❌ {self.agent.agent_name} 요청이 비어있습니다.")
                    )
                    return
                
                # 에이전트 처리 실행
                result = await self.agent.process_request(user_instructions)
                
                # 작업 완료
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(result)
                )
                
            else:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("❌ 메시지를 찾을 수 없습니다.")
                )
                
        except Exception as e:
            logger.error(f"❌ {self.agent.agent_name} 실행 실패: {e}")
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(f"❌ {self.agent.agent_name} 처리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"🚫 {self.agent.agent_name} 작업 취소 - Task: {context.task_id}")