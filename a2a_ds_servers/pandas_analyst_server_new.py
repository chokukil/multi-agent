#!/usr/bin/env python3
"""
PandasAnalyst Server - A2A SDK Complete Implementation with Langfuse Integration

Following the same pattern as SQL Database and MLflow Tools agents:
- AgentExecutor inheritance
- Complete Langfuse integration with 3-stage span structure
- TaskUpdater pattern
- A2A standard server initialization

포트: 8315
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import io
import json
import time
from typing import Dict, Any
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState, TextPart
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Langfuse 통합 모듈 임포트
try:
    from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse 통합 모듈 로드 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger.warning(f"⚠️ Langfuse 통합 모듈 로드 실패: {e}")


class PandasAIDataProcessor:
    """pandas-ai 스타일 데이터 프로세서"""
    
    def parse_data_from_message(self, user_instructions: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터 파싱"""
        logger.info("🔍 데이터 파싱 시작")
        
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
            json_pattern = r'\[.*?\]|\{.*?\}'
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
        
        logger.info("⚠️ 파싱 가능한 데이터 없음")
        return None


class PandasAnalystAgentExecutor(AgentExecutor):
    """
    PandasAnalyst Agent Executor with complete A2A and Langfuse integration
    
    Following the same pattern as SQL Database and MLflow Tools agents:
    - Inherits from AgentExecutor
    - Complete Langfuse integration with 3-stage span structure
    - TaskUpdater pattern for execution management
    """
    
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        
        # Initialize Langfuse tracer
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ PandasAnalyst Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        # Initialize PandasAnalyst wrapper
        try:
            from a2a_ds_servers.base.pandas_analyst_a2a_wrapper import PandasAnalystA2AWrapper
            self.pandas_wrapper = PandasAnalystA2AWrapper()
            logger.info("✅ PandasAnalyst A2A Wrapper 초기화 완료")
        except Exception as e:
            logger.error(f"❌ PandasAnalyst A2A Wrapper 초기화 실패: {e}")
            self.pandas_wrapper = None
        
        logger.info("🐼 PandasAnalyst AgentExecutor 초기화 완료")
        logger.info("🚀 LLM-First 동적 pandas 코드 생성 시스템 활성화")
        logger.info("🔧 8개 핵심 데이터 조작 기능 준비 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute method with TaskUpdater pattern and Langfuse integration"""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user message
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            logger.info(f"📥 Processing pandas analysis: {user_message[:100]}...")
            
            if not user_message:
                user_message = "Please provide pandas analysis instructions with data."
            
            # Create Langfuse session
            session_id = None
            if self.langfuse_tracer:
                session_id = self.langfuse_tracer.create_session(user_message)
            
            # Process with 3-stage Langfuse span structure
            result = await self._process_with_langfuse_spans(user_message, session_id)
            
            # Complete task with result
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=result)])
            )
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            # Report error through TaskUpdater
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"Pandas analysis failed: {str(e)}")]))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")
    
    async def _process_with_langfuse_spans(self, user_message: str, session_id: str) -> str:
        """Process pandas analysis with 3-stage Langfuse span structure"""
        start_time = datetime.now()
        
        # Stage 1: Request Parsing
        if self.langfuse_tracer:
            self.langfuse_tracer.add_span(
                name="request_parsing",
                input_data={"user_message": user_message},
                metadata={"stage": "parsing", "agent": "PandasAnalyst"},
                start_time=start_time
            )
        
        # Parse data from message
        df = self.data_processor.parse_data_from_message(user_message)
        
        parsing_end_time = datetime.now()
        if self.langfuse_tracer:
            self.langfuse_tracer.add_span(
                name="request_parsing",
                output_data={
                    "data_found": df is not None,
                    "data_shape": df.shape if df is not None else None
                },
                end_time=parsing_end_time
            )
        
        # Stage 2: Pandas Operations
        operations_start_time = datetime.now()
        if self.langfuse_tracer:
            self.langfuse_tracer.add_span(
                name="pandas_operations",
                input_data={
                    "has_data": df is not None,
                    "data_shape": df.shape if df is not None else None,
                    "user_instructions": user_message
                },
                metadata={"stage": "operations", "agent": "PandasAnalyst"},
                start_time=operations_start_time
            )
        
        # Process pandas analysis
        if df is None or df.empty:
            logger.info("📚 데이터 없음 - Pandas 가이드 제공")
            if self.pandas_wrapper:
                result = self.pandas_wrapper._generate_guidance(user_message)
            else:
                result = self._generate_fallback_guidance(user_message)
        else:
            logger.info(f"🐼 Processing pandas data: {df.shape}")
            if self.pandas_wrapper:
                wrapped_result = await self.pandas_wrapper.process_request(user_message)
                result = self._format_pandas_result(wrapped_result, df, user_message)
            else:
                result = self._generate_fallback_analysis(df, user_message)
        
        operations_end_time = datetime.now()
        if self.langfuse_tracer:
            self.langfuse_tracer.add_span(
                name="pandas_operations",
                output_data={
                    "result_length": len(result),
                    "processing_time": (operations_end_time - operations_start_time).total_seconds()
                },
                end_time=operations_end_time
            )
        
        # Stage 3: Save Results
        save_start_time = datetime.now()
        if self.langfuse_tracer:
            self.langfuse_tracer.add_span(
                name="save_results",
                input_data={"result_ready": True},
                metadata={"stage": "saving", "agent": "PandasAnalyst"},
                start_time=save_start_time
            )
        
        # Finalize result
        final_result = self._finalize_result(result, user_message, start_time)
        
        save_end_time = datetime.now()
        if self.langfuse_tracer:
            self.langfuse_tracer.add_span(
                name="save_results",
                output_data={
                    "final_result_length": len(final_result),
                    "total_time": (save_end_time - start_time).total_seconds()
                },
                end_time=save_end_time
            )
        
        return final_result
    
    def _format_pandas_result(self, wrapped_result: str, df: pd.DataFrame, user_message: str) -> str:
        """Format the pandas analysis result"""
        return f"""# 🐼 **PandasAnalyst Complete Analysis**

## 📝 **Request**
{user_message}

## 📊 **Data Information**
- **Shape**: {df.shape[0]:,} rows × {df.shape[1]:,} columns
- **Columns**: {', '.join(df.columns.tolist())}
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

## 🧠 **Analysis Results**
{wrapped_result}

## 🎯 **Available Functions**
1. **load_data_formats()** - Load various data formats
2. **inspect_data()** - Data structure and quality inspection
3. **select_data()** - Advanced data selection and filtering
4. **manipulate_data()** - Complex data transformation
5. **aggregate_data()** - Grouping and aggregation operations
6. **merge_data()** - Data joining and merging
7. **clean_data()** - Data cleaning and preprocessing
8. **perform_statistical_analysis()** - Statistical analysis

✅ **PandasAnalyst LLM-First analysis completed successfully!**
"""
    
    def _generate_fallback_guidance(self, user_message: str) -> str:
        """Generate fallback guidance when wrapper is not available"""
        return f"""# 🐼 **PandasAnalyst Guide**

## 📝 **Your Request**
{user_message}

## 🧠 **PandasAnalyst Capabilities**

I'm a specialized pandas data analysis agent that can help you with:

### 📊 **Data Operations**
- Load data from CSV, JSON, Excel formats
- Inspect data structure and quality
- Filter and select data with complex conditions
- Transform and manipulate data
- Perform grouping and aggregation
- Merge and join datasets
- Clean and preprocess data
- Generate statistical analysis

### 💡 **Usage Examples**
```text
With CSV data:
name,age,city
John,25,Seoul
Jane,30,Busan

Ask me to:
- "Analyze the age distribution"
- "Filter data for age > 25"
- "Group by city and calculate average age"
- "Clean and optimize the data"
```

### 🚀 **How to Use**
1. Provide your data in CSV or JSON format
2. Describe what analysis you want
3. I'll generate custom pandas code and execute it
4. Get comprehensive results and insights

✅ **Ready to analyze your data with advanced pandas operations!**
"""
    
    def _generate_fallback_analysis(self, df: pd.DataFrame, user_message: str) -> str:
        """Generate fallback analysis when wrapper is not available"""
        try:
            # Basic analysis
            basic_info = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Generate summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            return f"""# 🐼 **PandasAnalyst Fallback Analysis**

## 📝 **Request**
{user_message}

## 📊 **Data Overview**
- **Shape**: {basic_info['shape'][0]:,} rows × {basic_info['shape'][1]:,} columns
- **Memory Usage**: {basic_info['memory_usage_mb']:.2f} MB
- **Numeric Columns**: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})
- **Categorical Columns**: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})

## 📈 **Basic Statistics**
{df.describe().to_string() if len(numeric_cols) > 0 else 'No numeric columns for statistics'}

## 🔍 **Missing Values**
{', '.join([f"{col}: {count}" for col, count in basic_info['missing_values'].items() if count > 0]) or 'No missing values'}

## 💡 **Recommendations**
- Use specific pandas operations for deeper analysis
- Consider data cleaning if missing values exist
- Explore correlations between numeric variables
- Analyze distributions and outliers

✅ **Basic analysis completed. Provide specific instructions for advanced operations!**
"""
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return f"Data analysis failed: {str(e)}"
    
    def _finalize_result(self, result: str, user_message: str, start_time: datetime) -> str:
        """Finalize the result with timing information"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return f"""{result}

---
**Processing Time**: {processing_time:.2f} seconds | **Agent**: PandasAnalyst | **Port**: 8315
"""


def create_agent_card() -> AgentCard:
    """Create Agent Card for PandasAnalyst"""
    
    # Pandas Analysis skill definition
    pandas_skill = AgentSkill(
        id="pandas_data_analysis",
        name="Pandas Data Analysis",
        description="고급 pandas 라이브러리를 활용한 전문적인 데이터 분석, 조작, 변환, 및 처리 작업을 수행합니다. 동적 코드 생성을 통해 맞춤형 데이터 솔루션을 제공합니다.",
        tags=["pandas", "data-analysis", "data-manipulation", "data-processing", "python", "dataframe"],
        examples=[
            "데이터를 로드하고 기본 정보를 확인해주세요",
            "특정 조건에 맞는 데이터를 필터링해주세요",
            "그룹별로 데이터를 집계하고 통계를 계산해주세요",
            "데이터를 피벗 테이블로 변환해주세요",
            "결측값을 처리하고 데이터를 정제해주세요",
            "새로운 피처를 생성하고 데이터를 변환해주세요",
            "데이터를 다양한 형식으로 내보내주세요",
            "복잡한 데이터 조작 작업을 수행해주세요"
        ]
    )
    
    # Agent Card creation
    agent_card = AgentCard(
        name="PandasAnalyst",
        description="고급 pandas 라이브러리 전문가입니다. LLM-first 접근방식으로 동적 pandas 코드를 생성하여 복잡한 데이터 분석, 조작, 변환 작업을 실시간으로 수행합니다. 맞춤형 데이터 솔루션 제공에 특화되어 있습니다.",
        url="http://localhost:8315/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[pandas_skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    return agent_card


def main():
    """Main function to start the PandasAnalyst server following A2A standard pattern"""
    logger.info("🐼 PandasAnalyst A2A Server 시작중...")
    logger.info("📍 포트: 8315")
    logger.info("🔗 URL: http://localhost:8315/")
    logger.info("🚀 LLM-First 동적 pandas 코드 생성 시스템")
    logger.info("🔧 8개 핵심 데이터 조작 기능 준비 완료")
    logger.info("🎯 Langfuse 통합 및 TaskUpdater 패턴 적용")
    logger.info("="*80)
    
    # A2A application setup following standard pattern
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=PandasAnalystAgentExecutor(),
        agent_card=create_agent_card()
    )
    
    # Create Starlette application
    server = A2AStarletteApplication(
        request_handler=request_handler,
        task_store=task_store
    )
    
    # Build and run server
    server.build()
    
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8315,
        log_level="info"
    )


if __name__ == "__main__":
    main()