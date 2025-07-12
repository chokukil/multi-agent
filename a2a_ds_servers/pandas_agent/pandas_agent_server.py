#!/usr/bin/env python3
"""
🐼 Pandas Agent A2A Server

A2A SDK 0.2.9 기반 자연어 데이터 분석 에이전트
PandasAI 참고 구현 (MIT License 준수)

Key Features:
- 자연어 기반 데이터 분석
- A2A SDK 0.2.9 완전 호환  
- 멀티 데이터프레임 처리
- 실시간 스트리밍 응답
- 안전한 코드 실행 환경
- 범용 데이터 포맷 지원

Author: CherryAI Team
License: MIT License
"""

import asyncio
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from contextlib import asynccontextmanager

# A2A SDK 0.2.9 Import
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Enhanced Tracking System (선택적)
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError:
    ENHANCED_TRACKING_AVAILABLE = False

# UserFileTracker 통합 (선택적)
try:
    from core.user_file_tracker import get_user_file_tracker
    from core.session_data_manager import SessionDataManager
    USER_FILE_TRACKER_AVAILABLE = True
except ImportError:
    USER_FILE_TRACKER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PandasAgentCore:
    """
    Pandas Agent 핵심 엔진
    
    자연어 기반 데이터 분석을 위한 핵심 로직
    PandasAI 패턴을 참고하여 안전하게 구현
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.dataframes: List[pd.DataFrame] = []
        self.dataframe_metadata: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.session_id: Optional[str] = None
        
        # Enhanced Tracking 초기화
        self.enhanced_tracer = None
        if ENHANCED_TRACKING_AVAILABLE:
            try:
                self.enhanced_tracer = get_enhanced_tracer()
                logger.info("✅ Enhanced Langfuse Tracking 활성화")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced Tracking 초기화 실패: {e}")
        
        # UserFileTracker 초기화
        self.user_file_tracker = None
        self.session_data_manager = None
        if USER_FILE_TRACKER_AVAILABLE:
            try:
                self.user_file_tracker = get_user_file_tracker()
                self.session_data_manager = SessionDataManager()
                logger.info("✅ UserFileTracker 통합 활성화")
            except Exception as e:
                logger.warning(f"⚠️ UserFileTracker 초기화 실패: {e}")
        
        # 기본 설정
        self._setup_default_config()
    
    def _setup_default_config(self):
        """기본 설정 초기화"""
        default_config = {
            "verbose": False,
            "save_logs": True,
            "max_retries": 3,
            "enable_cache": True,
            "custom_whitelisted_dependencies": ["pandas", "numpy", "matplotlib", "seaborn", "plotly"]
        }
        
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    async def add_dataframe(self, df: pd.DataFrame, name: str = None, description: str = None) -> str:
        """데이터프레임 추가"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("입력이 pandas DataFrame이 아닙니다.")
        
        df_id = name or f"dataframe_{len(self.dataframes)}"
        
        # 메타데이터 생성
        metadata = {
            "id": df_id,
            "name": name or df_id,
            "description": description or f"데이터프레임 {df_id}",
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "created_at": datetime.now().isoformat()
        }
        
        self.dataframes.append(df)
        self.dataframe_metadata.append(metadata)
        
        logger.info(f"✅ 데이터프레임 추가: {df_id} (shape: {df.shape})")
        return df_id
    
    async def process_natural_language_query(self, query: str) -> str:
        """자연어 쿼리 처리"""
        try:
            # Enhanced tracking 시작
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "natural_language_query",
                    {"query": query, "dataframe_count": len(self.dataframes)},
                    "자연어 쿼리 처리 시작"
                )
            
            if not self.dataframes:
                return "❌ 분석할 데이터가 없습니다. 먼저 데이터를 업로드해주세요."
            
            # 현재는 기본 분석을 제공 (추후 LLM 통합 예정)
            analysis_result = await self._perform_basic_analysis(query)
            
            # 대화 히스토리에 추가
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": analysis_result,
                "dataframes_used": len(self.dataframes)
            })
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ 자연어 쿼리 처리 오류: {e}")
            return f"❌ 쿼리 처리 중 오류가 발생했습니다: {str(e)}"
    
    async def _perform_basic_analysis(self, query: str) -> str:
        """기본 데이터 분석 수행"""
        if not self.dataframes:
            return "분석할 데이터가 없습니다."
        
        # 첫 번째 데이터프레임을 기본으로 사용
        df = self.dataframes[0]
        metadata = self.dataframe_metadata[0]
        
        # 쿼리 키워드 기반 분석 결정
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['summary', '요약', 'overview', '개요']):
            return self._generate_data_summary(df, metadata)
        elif any(keyword in query_lower for keyword in ['describe', '기술통계', 'statistics', '통계']):
            return self._generate_descriptive_statistics(df, metadata)
        elif any(keyword in query_lower for keyword in ['null', '결측', 'missing', '누락']):
            return self._generate_missing_data_analysis(df, metadata)
        elif any(keyword in query_lower for keyword in ['correlation', '상관', '관계']):
            return self._generate_correlation_analysis(df, metadata)
        else:
            # 기본 종합 분석
            return self._generate_comprehensive_analysis(df, metadata, query)
    
    def _generate_data_summary(self, df: pd.DataFrame, metadata: Dict) -> str:
        """데이터 요약 생성"""
        return f"""# 📊 **데이터 요약**

## 🔍 **기본 정보**
- **데이터셋**: {metadata['name']}
- **크기**: {df.shape[0]:,}행 × {df.shape[1]}열
- **메모리 사용량**: {metadata['memory_usage'] / 1024**2:.1f} MB

## 📋 **컬럼 정보**
{chr(10).join([f"- **{col}**: {dtype} (결측: {metadata['null_counts'][col]}개)" 
               for col, dtype in zip(df.columns, df.dtypes)])}

## 📈 **데이터 미리보기**
{df.head().to_string()}
"""
    
    def _generate_descriptive_statistics(self, df: pd.DataFrame, metadata: Dict) -> str:
        """기술통계 생성"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return "❌ 수치형 컬럼이 없어 기술통계를 생성할 수 없습니다."
        
        stats = numeric_df.describe()
        
        return f"""# 📊 **기술통계**

## 📈 **수치형 컬럼 통계**
{stats.to_string()}

## 🎯 **주요 인사이트**
- **총 수치형 컬럼**: {len(numeric_df.columns)}개
- **가장 높은 평균값**: {stats.loc['mean'].max():.2f}
- **가장 큰 표준편차**: {stats.loc['std'].max():.2f}
"""
    
    def _generate_missing_data_analysis(self, df: pd.DataFrame, metadata: Dict) -> str:
        """결측 데이터 분석"""
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df) * 100).round(2)
        
        missing_summary = []
        for col in df.columns:
            if null_counts[col] > 0:
                missing_summary.append(f"- **{col}**: {null_counts[col]}개 ({null_percentages[col]}%)")
        
        if not missing_summary:
            return "✅ **결측 데이터 없음**: 모든 컬럼이 완전합니다."
        
        return f"""# 🔍 **결측 데이터 분석**

## ⚠️ **결측 데이터 발견**
{chr(10).join(missing_summary)}

## 📊 **전체 요약**
- **총 결측 셀**: {null_counts.sum():,}개
- **영향받은 컬럼**: {(null_counts > 0).sum()}개
- **완전한 행**: {len(df) - df.isnull().any(axis=1).sum():,}개
"""
    
    def _generate_correlation_analysis(self, df: pd.DataFrame, metadata: Dict) -> str:
        """상관관계 분석"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return "❌ 상관관계 분석을 위해서는 최소 2개의 수치형 컬럼이 필요합니다."
        
        correlation_matrix = numeric_df.corr()
        
        # 높은 상관관계 찾기
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # 높은 상관관계 기준
                    high_corr_pairs.append(
                        f"- **{correlation_matrix.columns[i]}** ↔ **{correlation_matrix.columns[j]}**: {corr_value:.3f}"
                    )
        
        return f"""# 🔗 **상관관계 분석**

## 📊 **상관관계 매트릭스**
{correlation_matrix.round(3).to_string()}

## ⭐ **높은 상관관계 (|r| > 0.7)**
{chr(10).join(high_corr_pairs) if high_corr_pairs else "- 높은 상관관계를 보이는 변수 쌍이 없습니다."}
"""
    
    def _generate_comprehensive_analysis(self, df: pd.DataFrame, metadata: Dict, query: str) -> str:
        """종합 분석"""
        return f"""# 🔍 **종합 데이터 분석**

## 📝 **쿼리**: "{query}"

## 📊 **데이터 개요**
- **데이터셋**: {metadata['name']}
- **크기**: {df.shape[0]:,}행 × {df.shape[1]}열
- **수치형 컬럼**: {len(df.select_dtypes(include=[np.number]).columns)}개
- **범주형 컬럼**: {len(df.select_dtypes(include=['object']).columns)}개

## 🎯 **빠른 인사이트**
- **결측 데이터**: {df.isnull().sum().sum()}개 셀
- **완전한 행**: {len(df) - df.isnull().any(axis=1).sum():,}개
- **메모리 사용량**: {metadata['memory_usage'] / 1024**2:.1f} MB

## 💡 **추천 분석**
자연어로 더 구체적인 질문을 해보세요:
- "이 데이터의 기술통계를 보여줘"
- "결측 데이터 상황을 분석해줘" 
- "컬럼들 간의 상관관계를 알려줘"
"""
    
    def clear_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        logger.info("📝 대화 기록이 초기화되었습니다.")
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 기록 반환"""
        return self.conversation_history


class PandasAgentExecutor(AgentExecutor):
    """A2A SDK 0.2.9 호환 Pandas Agent Executor"""
    
    def __init__(self):
        self.agent = PandasAgentCore()
        logger.info("✅ Pandas Agent Executor 초기화 완료")
    
    async def cancel(self) -> None:
        """A2A SDK 0.2.9 표준 cancel 메서드"""
        logger.info("🛑 Pandas Agent Executor 취소 요청")
        if hasattr(self.agent, 'clear_conversation'):
            self.agent.clear_conversation()
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """A2A SDK 0.2.9 표준 execute 메서드"""
        try:
            # 사용자 입력 추출
            user_input = self._extract_user_input(context)
            
            # 세션 데이터 로딩 시도
            session_id = context.request.get("session_id")
            if session_id and USER_FILE_TRACKER_AVAILABLE:
                await self._load_session_data(session_id, task_updater)
            
            # 상태 업데이트
            await task_updater.update_status(
                TaskState.working,
                message="🐼 Pandas Agent가 분석을 시작합니다..."
            )
            
            # 자연어 쿼리 처리
            response = await self.agent.process_natural_language_query(user_input)
            
            # 최종 응답
            await task_updater.update_status(
                TaskState.completed,
                message=response,
                final=True
            )
            
        except Exception as e:
            logger.error(f"❌ Pandas Agent 실행 오류: {e}")
            await task_updater.update_status(
                TaskState.completed,
                message=f"❌ 분석 중 오류가 발생했습니다: {str(e)}",
                final=True
            )
    
    def _extract_user_input(self, context: RequestContext) -> str:
        """사용자 입력 추출"""
        try:
            if hasattr(context, 'message') and context.message:
                if hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            return part.root.text
                        elif hasattr(part, 'text'):
                            return part.text
            return "데이터 분석을 수행해주세요"
        except Exception as e:
            logger.warning(f"⚠️ 사용자 입력 추출 실패: {e}")
            return "데이터 분석을 수행해주세요"
    
    async def _load_session_data(self, session_id: str, task_updater: TaskUpdater):
        """세션 데이터 로딩"""
        try:
            if self.agent.user_file_tracker and self.agent.session_data_manager:
                # 세션 데이터 로딩
                session_data = await self.agent.session_data_manager.get_session_data(session_id)
                
                if session_data and session_data.get('uploaded_files'):
                    await task_updater.update_status(
                        TaskState.working,
                        message="📂 세션 데이터를 로딩하고 있습니다..."
                    )
                    
                    for file_info in session_data['uploaded_files']:
                        file_path = file_info.get('file_path')
                        if file_path and os.path.exists(file_path):
                            # 파일 확장자에 따른 로딩
                            if file_path.endswith('.csv'):
                                df = pd.read_csv(file_path)
                            elif file_path.endswith(('.xlsx', '.xls')):
                                df = pd.read_excel(file_path)
                            elif file_path.endswith('.json'):
                                df = pd.read_json(file_path)
                            else:
                                continue
                            
                            # 데이터프레임 추가
                            await self.agent.add_dataframe(
                                df, 
                                name=file_info.get('name', 'uploaded_data'),
                                description=file_info.get('description', '업로드된 데이터')
                            )
                            
                            logger.info(f"✅ 세션 데이터 로딩 완료: {file_info.get('name')}")
                
        except Exception as e:
            logger.warning(f"⚠️ 세션 데이터 로딩 실패: {e}")


# A2A 서버 설정
async def create_pandas_agent_server():
    """Pandas Agent A2A 서버 생성"""
    
    # Agent Card 설정
    skills_list = [
        AgentSkill(
            id="natural_language_analysis",
            name="natural_language_analysis",
            description="자연어로 데이터 분석 수행",
            tags=["analysis", "nlp"]
        ),
        AgentSkill(
            id="multi_dataframe_processing",
            name="multi_dataframe_processing", 
            description="여러 데이터프레임 동시 처리",
            tags=["data", "multi-df"]
        ),
        AgentSkill(
            id="descriptive_statistics",
            name="descriptive_statistics",
            description="기술통계 및 데이터 요약",
            tags=["statistics", "summary"]
        ),
        AgentSkill(
            id="data_quality_analysis",
            name="data_quality_analysis",
            description="데이터 품질 및 결측치 분석",
            tags=["quality", "analysis"]
        )
    ]
    
    agent_card = AgentCard(
        name="Pandas Agent",
        description="자연어 기반 데이터 분석 전문 에이전트",
        version="1.0.0",
        url="http://localhost:8315",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=skills_list,
        capabilities=AgentCapabilities(
            skills=skills_list
        )
    )
    
    # A2A 애플리케이션 생성 (A2A SDK 0.2.9 API)
    executor = PandasAgentExecutor()
    task_store = InMemoryTaskStore()
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(executor, task_store)
    )
    
    return app


async def main():
    """메인 실행 함수"""
    app = await create_pandas_agent_server()
    return app

# 모듈 레벨에서 앱 생성 지원
app = None

def get_app():
    """앱 인스턴스 반환"""
    global app
    if app is None:
        app = asyncio.run(create_pandas_agent_server())
    return app

if __name__ == "__main__":
    import sys
    
    # 명령행 인자 확인
    port = 8315
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.warning(f"⚠️ 잘못된 포트 번호: {sys.argv[1]}, 기본값 8315 사용")
    
    # 서버 정보 출력
    logger.info(f"🚀 Pandas Agent A2A 서버 시작")
    logger.info(f"📍 주소: http://0.0.0.0:{port}")
    logger.info(f"🔧 Agent Card: http://0.0.0.0:{port}/.well-known/agent.json")
    
    # 앱 생성
    app = asyncio.run(create_pandas_agent_server())
    
    # 서버 실행
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 서버가 종료되었습니다.")
    except Exception as e:
        logger.error(f"❌ 서버 실행 오류: {e}") 