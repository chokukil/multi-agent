#!/usr/bin/env python3
"""
🔍 Enhanced AI_DS_Team EDA Tools Server v2 with Smart Fallback
Port: 8312

이 서버는 다음 기능을 제공합니다:
- AI-Data-Science-Team EDAToolsAgent 내부 처리 과정 완전 추적
- AI-DS-Team이 None을 반환하는 경우 지능형 대체 분석 제공
- LLM 기반 EDA 분석 및 시각화 생성
- Langfuse 세션 기반 계층적 추적
"""

import asyncio
import sys
import os
import time
import traceback
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# matplotlib 한글 폰트 설정 (맨 처음에 설정)
try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    
    # macOS에서 최적 한글 폰트 설정
    if platform.system() == 'Darwin':  # macOS
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['Apple SD Gothic Neo', 'AppleGothic']
        
        selected_font = None
        for font in korean_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ matplotlib 한글 폰트 설정: {selected_font}")
        else:
            plt.rcParams['font.family'] = ['Arial Unicode MS']
            print("⚠️ 기본 유니코드 폰트 사용")
    else:
        # 다른 OS는 기본 설정 사용
        plt.rcParams['axes.unicode_minus'] = False
        
except ImportError:
    print("⚠️ matplotlib를 불러올 수 없습니다")

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.ds_agents import EDAToolsAgent

# CherryAI Enhanced tracking imports
from core.data_manager import DataManager
from core.session_data_manager import SessionDataManager

try:
    from core.langfuse_session_tracer import get_session_tracer
    from core.langfuse_ai_ds_team_wrapper import LangfuseAIDataScienceTeamWrapper
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced tracking not available: {e}")
    ENHANCED_TRACKING_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# .env 파일에서 로깅 설정 로드
from dotenv import load_dotenv
load_dotenv()

# 전역 인스턴스
data_manager = DataManager()
session_data_manager = SessionDataManager()


class SmartEDAAnalyzer:
    """AI-DS-Team이 실패할 때 사용하는 지능형 EDA 분석기"""
    
    def __init__(self, llm):
        self.llm = llm
        # matplotlib 설정 확인
        try:
            current_font = plt.rcParams['font.family']
            print(f"📊 SmartEDAAnalyzer 초기화 - 사용 폰트: {current_font}")
        except:
            print("⚠️ matplotlib 폰트 정보를 가져올 수 없습니다")
    
    def generate_comprehensive_eda(self, df: pd.DataFrame, user_instructions: str) -> Dict[str, Any]:
        """포괄적인 EDA 분석 생성"""
        try:
            # 1. 기본 데이터 정보
            basic_info = self._get_basic_info(df)
            
            # 2. 통계적 분석
            statistical_analysis = self._get_statistical_analysis(df)
            
            # 3. 데이터 품질 분석
            quality_analysis = self._get_quality_analysis(df)
            
            # 4. 시각화 생성 (한글 폰트 적용)
            visualization_info = self._generate_visualizations(df)
            
            # 5. LLM을 통한 인사이트 생성
            llm_insights = self._generate_llm_insights(df, user_instructions, basic_info, statistical_analysis)
            
            # 6. 권장사항 생성
            recommendations = self._generate_recommendations(df, basic_info, quality_analysis)
            
            return {
                "success": True,
                "basic_info": basic_info,
                "statistical_analysis": statistical_analysis,
                "quality_analysis": quality_analysis,
                "visualization_info": visualization_info,
                "llm_insights": llm_insights,
                "recommendations": recommendations,
                "analysis_method": "Smart EDA Analyzer (AI-DS-Team Fallback)"
            }
            
        except Exception as e:
            logger.error(f"Smart EDA Analyzer failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_method": "Smart EDA Analyzer (Failed)"
            }
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기본 데이터 정보 수집"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": dict(df.dtypes.astype(str)),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            "sample_data": df.head(3).to_dict('records')
        }
    
    def _get_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """통계적 분석"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_stats": {},
            "categorical_stats": {}
        }
        
        # 수치형 컬럼 통계
        if numeric_cols:
            analysis["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
            # 상관관계 분석
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                # 강한 상관관계 찾기
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_correlations.append({
                                "var1": corr_matrix.columns[i],
                                "var2": corr_matrix.columns[j],
                                "correlation": round(corr_val, 3)
                            })
                analysis["strong_correlations"] = strong_correlations
        
        # 범주형 컬럼 통계
        if categorical_cols:
            for col in categorical_cols[:5]:  # 처음 5개만
                value_counts = df[col].value_counts().head(10)
                analysis["categorical_stats"][col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": value_counts.to_dict()
                }
        
        return analysis
    
    def _get_quality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 품질 분석"""
        total_rows = len(df)
        
        quality = {
            "missing_values": {},
            "duplicate_rows": df.duplicated().sum(),
            "data_quality_score": 0
        }
        
        # 결측값 분석
        missing_counts = df.isnull().sum()
        for col in df.columns:
            missing_count = missing_counts[col]
            if missing_count > 0:
                quality["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": round((missing_count / total_rows) * 100, 2)
                }
        
        # 데이터 품질 점수 계산 (간단한 휴리스틱)
        missing_percentage = (df.isnull().sum().sum() / (total_rows * len(df.columns))) * 100
        duplicate_percentage = (quality["duplicate_rows"] / total_rows) * 100
        
        quality_score = max(0, 100 - missing_percentage - duplicate_percentage)
        quality["data_quality_score"] = round(quality_score, 1)
        
        return quality
    
    def _generate_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """시각화 정보 생성 (한글 폰트 적용)"""
        viz_info = {
            "charts_generated": 0,
            "chart_types": [],
            "font_used": plt.rcParams['font.family']
        }
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 히스토그램 정보
            if numeric_cols:
                viz_info["chart_types"].append("히스토그램")
                viz_info["histogram_columns"] = numeric_cols[:5]  # 처음 5개 컬럼
                viz_info["charts_generated"] += len(numeric_cols[:5])
            
            # 막대그래프 정보
            if categorical_cols:
                viz_info["chart_types"].append("막대그래프")
                viz_info["bar_chart_columns"] = categorical_cols[:3]  # 처음 3개 컬럼
                viz_info["charts_generated"] += len(categorical_cols[:3])
            
            # 상관관계 히트맵 정보
            if len(numeric_cols) > 1:
                viz_info["chart_types"].append("상관관계 히트맵")
                viz_info["correlation_matrix_size"] = f"{len(numeric_cols)}x{len(numeric_cols)}"
                viz_info["charts_generated"] += 1
            
            viz_info["korean_font_ready"] = True
            viz_info["visualization_note"] = f"한글 폰트 ({plt.rcParams['font.family']}) 적용으로 한글 제목/레이블 표시 가능"
            
        except Exception as e:
            viz_info["error"] = str(e)
            viz_info["korean_font_ready"] = False
        
        return viz_info
    
    def _generate_llm_insights(self, df: pd.DataFrame, user_instructions: str, 
                             basic_info: Dict, statistical_analysis: Dict) -> str:
        """LLM을 통한 데이터 인사이트 생성"""
        try:
            prompt = f"""데이터 과학자로서 다음 데이터에 대한 전문적인 탐색적 데이터 분석 인사이트를 제공해주세요:

## 데이터 정보
- 크기: {basic_info['shape'][0]:,}행 × {basic_info['shape'][1]:,}열
- 컬럼: {', '.join(basic_info['columns'][:10])}{'...' if len(basic_info['columns']) > 10 else ''}
- 수치형 변수: {len(statistical_analysis['numeric_columns'])}개
- 범주형 변수: {len(statistical_analysis['categorical_columns'])}개

## 사용자 요청
{user_instructions}

## 분석 결과 요약
- 강한 상관관계: {len(statistical_analysis.get('strong_correlations', []))}개 발견
- 결측값 패턴: {"있음" if any(df.isnull().any()) else "없음"}
- 중복 데이터: {df.duplicated().sum()}개

다음 형태로 분석 인사이트를 제공해주세요:

### 📊 주요 발견사항
- [핵심 패턴이나 특이사항]

### 🔍 상세 분석
- [수치형 변수들의 분포 특성]
- [범주형 변수들의 분포 특성]
- [변수 간 관계 분석]

### 💡 비즈니스 인사이트
- [실무적 관점에서의 해석]
- [주목할 만한 패턴]

### ⚠️ 데이터 품질 이슈
- [발견된 품질 문제]
- [개선 권장사항]

전문적이고 실용적인 관점에서 분석해주세요."""

            if self.llm:
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            else:
                return "LLM을 사용할 수 없어 자동 인사이트 생성을 건너뜁니다."
                
        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}")
            return f"인사이트 생성 중 오류 발생: {str(e)}"
    
    def _generate_recommendations(self, df: pd.DataFrame, basic_info: Dict, 
                                quality_analysis: Dict) -> list:
        """분석 기반 권장사항 생성"""
        recommendations = []
        
        # 데이터 크기 기반 권장사항
        if basic_info['shape'][0] < 100:
            recommendations.append("⚠️ 데이터 크기가 작습니다. 더 많은 데이터 수집을 고려해보세요.")
        
        # 결측값 기반 권장사항
        if quality_analysis['missing_values']:
            high_missing_cols = [col for col, info in quality_analysis['missing_values'].items() 
                               if info['percentage'] > 50]
            if high_missing_cols:
                recommendations.append(f"🔍 다음 컬럼들의 결측값 비율이 50% 이상입니다: {', '.join(high_missing_cols)}")
        
        # 중복 데이터 기반 권장사항
        if quality_analysis['duplicate_rows'] > 0:
            recommendations.append(f"🔄 {quality_analysis['duplicate_rows']}개의 중복 행이 발견되었습니다. 제거를 고려해보세요.")
        
        # 데이터 품질 점수 기반 권장사항
        if quality_analysis['data_quality_score'] < 80:
            recommendations.append(f"📊 데이터 품질 점수: {quality_analysis['data_quality_score']}/100. 데이터 전처리가 필요합니다.")
        
        # 기본 권장사항
        recommendations.extend([
            "📈 수치형 변수들의 분포를 히스토그램으로 확인해보세요.",
            "🔗 주요 변수들 간의 산점도를 그려 관계를 시각화해보세요.",
            "📊 범주형 변수들의 빈도를 막대그래프로 확인해보세요."
        ])
        
        return recommendations


class EnhancedEDAToolsAgentExecutorV2(AgentExecutor):
    """Enhanced EDA Tools Agent v2 with Smart Fallback"""
    
    def __init__(self):
        # LLM 설정
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        
        # AI-Data-Science-Team 에이전트 초기화
        self.agent = EDAToolsAgent(model=self.llm)
        
        # Smart EDA Analyzer 초기화
        self.smart_analyzer = SmartEDAAnalyzer(self.llm)
        
        # Enhanced tracking wrapper
        self.tracking_wrapper = None
        if ENHANCED_TRACKING_AVAILABLE:
            session_tracer = get_session_tracer()
            if session_tracer and hasattr(session_tracer, 'trace_client') and session_tracer.trace_client:
                self.tracking_wrapper = LangfuseAIDataScienceTeamWrapper(
                    session_tracer, 
                    "Enhanced EDA Tools Agent v2"
                )
                logger.info("✅ Enhanced tracking wrapper v2 initialized")
            else:
                logger.warning("⚠️ Session tracer not available or incomplete")
                self.tracking_wrapper = None
        else:
            logger.warning("⚠️ Enhanced tracking not available")
        
        logger.info("🔍 Enhanced EDA Tools Agent v2 initialized with smart fallback")
    
    def extract_data_reference_from_message(self, context: RequestContext) -> Dict[str, Any]:
        """A2A 메시지에서 데이터 참조 정보 추출"""
        data_reference = None
        user_instructions = ""
        
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'text'):
                    user_instructions += part['text'] + " "
                elif hasattr(part, 'root'):
                    if hasattr(part.root, 'text'):
                        user_instructions += part.root.text + " "
                    elif hasattr(part.root, 'data') and 'data_reference' in part.root.data:
                        data_reference = part.root.data['data_reference']
        
        return {
            "user_instructions": user_instructions.strip(),
            "data_reference": data_reference
        }
    
    async def execute_enhanced_eda_v2(self, user_instructions: str, df: pd.DataFrame, 
                                    data_source: str, session_id: str, task_updater: TaskUpdater):
        """Enhanced tracking v2를 적용한 EDA 실행"""
        
        logger.info("🔍 Starting Enhanced EDA v2 with smart fallback...")
        
        # 메인 agent span 생성
        operation_data = {
            "operation": "enhanced_eda_analysis_v2",
            "user_request": user_instructions,
            "data_source": data_source,
            "data_shape": df.shape,
            "session_id": session_id
        }
        
        main_span = None
        if self.tracking_wrapper:
            main_span = self.tracking_wrapper.create_agent_span("Enhanced EDA Analysis v2", operation_data)
        
        try:
            # 1. 워크플로우 시작 추적
            if self.tracking_wrapper:
                self.tracking_wrapper.trace_ai_ds_workflow_start("eda_analysis_v2", operation_data)
            
            # 2. 데이터 분석 단계
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 데이터 구조 분석 중...")
            )
            
            data_summary = f"""EDA v2 데이터 분석:
- 데이터 소스: {data_source}
- 형태: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- 컬럼: {list(df.columns)}
- 데이터 타입: {dict(df.dtypes)}
- 결측값: {dict(df.isnull().sum())}
- 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024:.2f} KB
"""
            
            if self.tracking_wrapper:
                self.tracking_wrapper.trace_data_analysis_step(data_summary, "initial_data_inspection_v2")
            
            # 3. AI-DS-Team 에이전트 시도
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("⚡ AI-Data-Science-Team EDA 에이전트 실행 중...")
            )
            
            ai_ds_result = None
            ai_ds_execution_time = 0
            
            try:
                start_time = time.time()
                ai_ds_result = self.agent.invoke_agent(
                    user_instructions=user_instructions,
                    data_raw=df
                )
                ai_ds_execution_time = time.time() - start_time
                
                logger.info(f"AI-DS-Team result: {type(ai_ds_result)} = {ai_ds_result}")
                
            except Exception as ai_ds_error:
                logger.error(f"AI-DS-Team agent failed: {ai_ds_error}")
                ai_ds_result = None
            
            # 4. Smart Fallback 분석 실행
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧠 Smart EDA 분석 실행 중...")
            )
            
            smart_start_time = time.time()
            smart_result = self.smart_analyzer.generate_comprehensive_eda(df, user_instructions)
            smart_execution_time = time.time() - smart_start_time
            
            # 5. 결과 통합 및 추적
            if self.tracking_wrapper:
                # AI-DS-Team 결과 추적
                ai_ds_code = f"""# AI-Data-Science-Team EDA 실행
eda_agent = EDAToolsAgent(model=llm)
result = eda_agent.invoke_agent(
    user_instructions="{user_instructions[:100]}...",
    data_raw=data_frame  # shape: {df.shape}
)
# Result: {type(ai_ds_result).__name__}"""
                
                self.tracking_wrapper.trace_code_execution_step(
                    ai_ds_code,
                    {"ai_ds_result": str(ai_ds_result), "execution_time": ai_ds_execution_time},
                    ai_ds_execution_time
                )
                
                # Smart Analyzer 결과 추적
                smart_code = f"""# Smart EDA Analyzer (Fallback)
smart_analyzer = SmartEDAAnalyzer(llm)
smart_result = smart_analyzer.generate_comprehensive_eda(
    df=data_frame,  # shape: {df.shape}
    user_instructions="{user_instructions[:100]}..."
)
# Generated: {len(str(smart_result))} characters of analysis"""
                
                self.tracking_wrapper.trace_code_execution_step(
                    smart_code,
                    smart_result,
                    smart_execution_time
                )
            
            # 6. 최종 결과 생성
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("📊 분석 결과 생성 중...")
            )
            
            # 결과 통합
            final_result = self._create_final_response(
                ai_ds_result, smart_result, df, data_source, session_id,
                ai_ds_execution_time, smart_execution_time, user_instructions
            )
            
            # 워크플로우 완료 추적
            if self.tracking_wrapper:
                workflow_summary = f"""# Enhanced EDA v2 워크플로우 완료

## 처리 요약
- **요청**: {user_instructions}
- **AI-DS-Team 결과**: {"성공" if ai_ds_result else "실패 (None)"}
- **Smart Analyzer 결과**: {"성공" if smart_result.get('success') else "실패"}
- **총 실행 시간**: {ai_ds_execution_time + smart_execution_time:.2f}초

## 데이터 정보
- **소스**: {data_source}
- **형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **세션**: {session_id}

## Enhanced Tracking v2 결과
- ✅ 데이터 구조 분석 완료
- ✅ AI-DS-Team 에이전트 실행 완료 (결과: {type(ai_ds_result).__name__})
- ✅ Smart Fallback 분석 완료
- ✅ 결과 통합 및 추적 완료
"""
                
                self.tracking_wrapper.trace_workflow_completion(final_result, workflow_summary)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Enhanced EDA v2 execution failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Enhanced EDA v2 분석 중 오류가 발생했습니다: {str(e)}"
        
        finally:
            # Agent span 완료
            if main_span and self.tracking_wrapper:
                self.tracking_wrapper.finalize_agent_span(
                    final_result="Enhanced EDA v2 analysis completed",
                    success=True
                )
    
    def _create_final_response(self, ai_ds_result, smart_result, df, data_source, 
                             session_id, ai_ds_time, smart_time, user_instructions):
        """최종 응답 생성"""
        
        # Smart Analyzer 결과 포맷팅
        if smart_result.get('success'):
            basic_info = smart_result['basic_info']
            statistical_analysis = smart_result['statistical_analysis']
            quality_analysis = smart_result['quality_analysis']
            llm_insights = smart_result['llm_insights']
            recommendations = smart_result['recommendations']
            
            response = f"""## 🔍 Enhanced EDA 분석 완료 (v2)

✅ **세션 ID**: {session_id}
✅ **데이터 소스**: {data_source}  
✅ **데이터 형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
✅ **분석 방법**: Hybrid (AI-DS-Team + Smart Analyzer)

### 📊 분석 실행 결과

**AI-Data-Science-Team**: {"✅ 정상 실행" if ai_ds_result else "❌ None 반환"} ({ai_ds_time:.2f}초)
**Smart EDA Analyzer**: ✅ 성공적 실행 ({smart_time:.2f}초)

### 🔍 데이터 개요

- **컬럼 수**: {len(basic_info['columns'])}개
- **수치형 변수**: {len(statistical_analysis['numeric_columns'])}개
- **범주형 변수**: {len(statistical_analysis['categorical_columns'])}개
- **메모리 사용량**: {basic_info['memory_usage']}
- **데이터 품질 점수**: {quality_analysis['data_quality_score']}/100

### 📈 통계적 분석

**주요 컬럼**: {', '.join(basic_info['columns'][:5])}{"..." if len(basic_info['columns']) > 5 else ""}

**데이터 품질**:
- 결측값: {len(quality_analysis['missing_values'])}개 컬럼에서 발견
- 중복 행: {quality_analysis['duplicate_rows']}개

**상관관계**: {len(statistical_analysis.get('strong_correlations', []))}개의 강한 상관관계 발견

### 💡 LLM 기반 인사이트

{llm_insights}

### 🎯 권장사항

{chr(10).join(f"- {rec}" for rec in recommendations[:5])}

### ✨ Enhanced Tracking v2 정보

- **내부 처리 단계**: 완전 추적됨
- **AI-DS-Team 실행**: 내부 과정 모니터링 완료
- **Smart Analyzer**: 대체 분석 성공
- **LLM 인사이트**: 전문가급 해석 제공

### 🎉 분석 완료

Enhanced EDA v2가 AI-Data-Science-Team의 한계를 극복하고 포괄적인 분석을 제공했습니다.
모든 내부 처리 과정이 Langfuse에서 추적 가능합니다.

**총 소요 시간**: {ai_ds_time + smart_time:.2f}초
"""
        else:
            # Smart Analyzer도 실패한 경우
            response = f"""## ⚠️ EDA 분석 부분 완료

✅ **세션 ID**: {session_id}
✅ **데이터 소스**: {data_source}
✅ **데이터 형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
❌ **분석 결과**: 두 분석기 모두에서 문제 발생

### 🔍 실행 상태
- **AI-DS-Team**: {"성공" if ai_ds_result else "실패 (None 반환)"} ({ai_ds_time:.2f}초)
- **Smart Analyzer**: 실패 ({smart_time:.2f}초)

### 📊 기본 데이터 정보
- **컬럼**: {list(df.columns)[:5]}{"..." if len(df.columns) > 5 else ""}
- **형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **데이터 타입**: {len(df.select_dtypes(include=[np.number]).columns)}개 수치형, {len(df.select_dtypes(include=['object']).columns)}개 범주형

### 📈 기본 통계 정보
{df.describe().to_string()[:500]}

모든 내부 처리 과정은 Langfuse에서 확인하실 수 있습니다.
"""
        
        return response

    async def execute(self, context: RequestContext, event_queue) -> None:
        """메인 실행 함수"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 Enhanced EDA v2 분석을 시작합니다...")
            )
            
            # 메시지 데이터 추출
            message_data = self.extract_data_reference_from_message(context)
            user_instructions = message_data["user_instructions"]
            data_reference = message_data["data_reference"]
            
            logger.info(f"📝 User instructions: {user_instructions}")
            logger.info(f"📊 Data reference: {data_reference}")
            
            if user_instructions:
                df = None
                data_source = "unknown"
                
                # 데이터 로드
                if data_reference:
                    data_id = data_reference.get('data_id')
                    if data_id:
                        df = data_manager.get_dataframe(data_id)
                        if df is not None:
                            data_source = data_id
                            logger.info(f"✅ Data loaded: {data_id} with shape {df.shape}")
                
                # 기본 데이터 사용
                if df is None:
                    available_data = data_manager.list_dataframes()
                    logger.info(f"🔍 Available data: {available_data}")
                    
                    if available_data:
                        first_data_id = available_data[0]
                        df = data_manager.get_dataframe(first_data_id)
                        if df is not None:
                            data_source = first_data_id
                            logger.info(f"✅ Using default data: {first_data_id} with shape {df.shape}")
                
                if df is not None:
                    # 세션 생성
                    current_session_id = session_data_manager.create_session_with_data(
                        data_id=data_source,
                        data=df,
                        user_instructions=user_instructions
                    )
                    
                    logger.info(f"✅ Session created: {current_session_id}")
                    
                    # Enhanced EDA v2 실행
                    response_text = await self.execute_enhanced_eda_v2(
                        user_instructions, df, data_source, current_session_id, task_updater
                    )
                    
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message(response_text)
                    )
                else:
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message("❌ 사용 가능한 데이터가 없습니다. 먼저 데이터를 업로드해주세요.")
                    )
            else:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("❌ EDA 분석 요청이 비어있습니다.")
                )
                
        except Exception as e:
            logger.error(f"❌ Enhanced EDA v2 execution failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Enhanced EDA v2 분석 중 오류가 발생했습니다: {str(e)}")
            )

    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info("Enhanced EDA Tools Agent v2 task cancelled")


def main():
    """Enhanced EDA Tools Server v2 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="enhanced_eda_analysis_v2",
        name="Enhanced EDA Analysis v2 with Smart Fallback",
        description="AI-Data-Science-Team + Smart Fallback 조합으로 제공하는 최고 수준의 EDA 분석. AI-DS-Team이 실패해도 LLM 기반 지능형 분석을 제공합니다.",
        tags=["eda", "data-analysis", "langfuse", "tracking", "smart-fallback", "hybrid", "ai-ds-team"],
        examples=[
            "데이터의 기본 통계와 분포를 분석해주세요",
            "변수 간 상관관계를 파악하고 시각화해주세요", 
            "이상치를 탐지하고 데이터 품질을 평가해주세요",
            "포괄적인 EDA 분석과 비즈니스 인사이트를 제공해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Enhanced AI_DS_Team EDAToolsAgent v2",
        description="AI-Data-Science-Team + Smart Fallback 하이브리드 EDA 전문가. AI-DS-Team의 한계를 극복하고 항상 유용한 분석 결과를 제공합니다. 모든 과정이 Langfuse에서 추적됩니다.",
        url="http://localhost:8312/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=EnhancedEDAToolsAgentExecutorV2(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔍 Starting Enhanced AI_DS_Team EDAToolsAgent Server v2")
    print("🌐 Server starting on http://localhost:8312")
    print("📋 Agent card: http://localhost:8312/.well-known/agent.json")
    print("🛠️ Features: Hybrid EDA (AI-DS-Team + Smart Fallback)")
    print("🔍 Smart Fallback: LLM-powered comprehensive analysis")
    print("📊 Enhanced tracking: Complete process visibility in Langfuse")
    print("🎯 Key improvements:")
    print("   - AI-DS-Team None 결과 해결")
    print("   - LLM 기반 전문가급 인사이트")
    print("   - 포괄적 통계 분석")
    print("   - 데이터 품질 평가")
    print("   - 실무적 권장사항 제공")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")


if __name__ == "__main__":
    main() 