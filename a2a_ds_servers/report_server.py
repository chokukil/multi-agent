import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python3
"""

Report Generator Server - A2A Compatible 
🎯 보고서 생성 기능 구현 (성공 패턴 적용)
포트: 8326
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
    # ReportGeneratorAgent는 존재하지 않으므로 기본 에이전트 사용
    ReportGeneratorAgent = None
except ImportError:
    ReportGeneratorAgent = None

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
                'title': [f'Report {i}' for i in range(1, 6)],
                'content': [f'This is report content {i}' for i in range(1, 6)],
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

class EnhancedReportGeneratorAgent:
    """Enhanced Report Generator Agent - 실제 보고서 생성 구현"""
    
    def __init__(self):
        logger.info("✅ Enhanced Report Generator Agent initialized")
        
    async def generate_report(self, df: pd.DataFrame, user_instructions: str) -> dict:
        """보고서 생성 처리 (성공 패턴)"""
        try:
            logger.info(f"📊 보고서 생성 시작: {df.shape}")
            
            # 기본 보고서 생성
            report_sections = self._create_report_sections(df, user_instructions)
            
            # 보고서 요약
            report_summary = self._generate_report_summary(df, report_sections)
            
            return {
                'report_sections': report_sections,
                'report_summary': report_summary,
                'user_instructions': user_instructions,
                'generated_at': datetime.now().isoformat()
            }
                    
            except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")
            raise
    
    def _create_report_sections(self, df: pd.DataFrame, instructions: str) -> list:
        """보고서 섹션 생성"""
        sections = []
        
        # 1. 실행 요약
        sections.append({
            'title': '📋 실행 요약',
            'content': f'데이터 분석 보고서가 생성되었습니다. 총 {len(df)}개의 레코드를 분석했습니다.',
            'type': 'summary'
        })
        
        # 2. 데이터 개요
        sections.append({
            'title': '📊 데이터 개요',
            'content': f'데이터 크기: {df.shape[0]}행 x {df.shape[1]}열\n컬럼: {", ".join(df.columns.tolist())}',
            'type': 'overview'
        })
        
        # 3. 기술 통계
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats_content = f'수치형 컬럼 ({len(numeric_cols)}개): {", ".join(numeric_cols.tolist())}'
            sections.append({
                'title': '📈 기술 통계',
                'content': stats_content,
                'type': 'statistics'
            })
        
        # 4. 요청 분석
        sections.append({
            'title': '🎯 요청 분석',
            'content': f'사용자 요청: {instructions}',
            'type': 'request'
        })
        
        return sections
    
    def _generate_report_summary(self, df: pd.DataFrame, sections: list) -> dict:
        """보고서 요약 생성"""
        return {
            'total_sections': len(sections),
            'data_size': df.shape,
            'columns_count': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }

class ReportGeneratorExecutor(AgentExecutor):
    """Report Generator A2A Executor (성공 패턴)"""
    
    def __init__(self):
        # 성공 패턴: 데이터 프로세서와 에이전트 초기화
        self.data_processor = PandasAIDataProcessor()
        self.agent = EnhancedReportGeneratorAgent()
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """실행 메서드 (성공 패턴)"""
        # 성공 패턴: TaskUpdater 올바른 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 성공 패턴: 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("Report Generator 작업을 시작합니다...")
            )
            
            # 성공 패턴: 메시지 추출
            user_message = ""
                for part in context.message.parts:
                    if part.root.kind == "text":
                    user_message += part.root.text
            
            logger.info(f"📥 Processing report generation query: {user_message}")
            
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
            logger.error(f"Report Generator 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"처리 중 오류 발생: {str(e)}")
            )
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """보고서 생성 처리 (성공 패턴)"""
        try:
            # 성공 패턴: 에이전트 호출
            report_result = await self.agent.generate_report(df, user_instructions)
            
            # 성공 패턴: 결과 검증 및 포맷팅
            if report_result and 'report_sections' in report_result:
                return self._generate_response(report_result, user_instructions)
            else:
                return self._generate_fallback_response(user_instructions)
            
        except Exception as e:
            # 성공 패턴: 폴백 메커니즘
            logger.warning(f"보고서 생성 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
    
    def _generate_response(self, report_result: dict, user_instructions: str) -> str:
        """보고서 결과 응답 생성 (성공 패턴)"""
        report_sections = report_result['report_sections']
        report_summary = report_result['report_summary']
        
        sections_content = "\n\n".join([
            f"## {section['title']}\n{section['content']}"
            for section in report_sections
        ])
        
        return f"""# 📊 **Report Generation Complete!**

## 📋 보고서 생성 결과

**생성된 섹션**: {report_summary['total_sections']}개
**데이터 크기**: {report_summary['data_size'][0]}행 x {report_summary['data_size'][1]}열
**컬럼 수**: {report_summary['columns_count']}개

## 📝 보고서 내용

{sections_content}

## 🎯 요청 내용
{user_instructions}

보고서가 성공적으로 생성되었습니다! 📊
"""
    
    def _generate_no_data_response(self, user_instructions: str) -> str:
        """데이터 없음 응답 (성공 패턴)"""
        return f"""# ❌ **보고서 생성할 데이터가 없습니다**

**해결 방법**:
1. **CSV 형태로 데이터 포함**
2. **JSON 형태로 데이터 포함**  
3. **샘플 데이터 요청**: "샘플 데이터로 보고서를 생성해주세요"

**요청**: {user_instructions}
"""
    
    def _generate_fallback_response(self, user_instructions: str) -> str:
        """폴백 응답 (성공 패턴)"""
        return f"""# ⚠️ **보고서 생성 처리 중 일시적 문제가 발생했습니다**

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
        id="report-generator",
        name="Report Generator Agent",
        description="데이터 분석 보고서 생성 기능",
        tags=["report", "generation", "analysis", "documentation"],
        examples=[
            "보고서를 생성해주세요",
            "분석 결과를 정리해주세요",
            "샘플 데이터로 보고서를 생성해주세요"
        ]
    )
    
    # 성공 패턴: Agent Card 정의
    agent_card = AgentCard(
        name="Report Generator Agent",
        description="Enhanced Report Generator Agent with comprehensive analysis capabilities",
        url="http://localhost:8326/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 성공 패턴: Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=ReportGeneratorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # 성공 패턴: A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"🚀 Starting Report Generator Server on http://localhost:8326")
    uvicorn.run(server.build(), host="0.0.0.0", port=8326, log_level="info")

if __name__ == "__main__":
    main()