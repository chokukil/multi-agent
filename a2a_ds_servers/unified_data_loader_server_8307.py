#!/usr/bin/env python3
"""
🍒 CherryAI 통합 데이터 로더 서버 (Unified Data Loader Server)
Port: 8307

pandas_agent 패턴을 기준으로 한 12개 A2A 에이전트 표준 데이터 로딩 시스템
- 100% 검증된 통합 인프라 사용
- LLM First 원칙 완전 준수
- UTF-8 인코딩 문제 완전 해결
- A2A SDK 0.2.9 표준 완벽 적용

Author: CherryAI Team
License: MIT License
"""

import uvicorn
import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

# 통합 데이터 시스템 imports (100% 검증된 인프라)
from unified_data_system import (
    UnifiedDataInterface,
    LLMFirstDataEngine,
    SmartDataFrame,
    DataProfile,
    QualityReport,
    CacheManager,
    EnhancedFileConnector
)
from unified_data_system.core.unified_data_interface import (
    DataIntent,
    DataIntentType,
    LoadingStrategy,
    A2AContext
)

logger = logging.getLogger(__name__)


class UnifiedDataLoaderExecutor(AgentExecutor):
    """
    통합 데이터 로더 실행기
    
    pandas_agent 패턴을 기준으로 한 완전히 검증된 통합 데이터 로딩 시스템
    - 12개 A2A 에이전트 표준 적용
    - LLM First 원칙 완전 준수
    - 100% 기능 보존, Mock 사용 금지
    """
    
    def __init__(self):
        """통합 데이터 로더 초기화"""
        super().__init__()
        
        # 핵심 컴포넌트 초기화
        self.cache_manager = CacheManager(max_size_mb=200, default_ttl=3600)
        self.llm_engine = LLMFirstDataEngine()
        self.file_connector = EnhancedFileConnector(self.cache_manager)
        
        # 로딩 통계
        self.stats = {
            "total_requests": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
            "average_load_time": 0.0
        }
        
        # 공유 저장소 경로
        self.shared_data_path = Path("a2a_ds_servers/artifacts/data/shared_dataframes")
        self.shared_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🍒 통합 데이터 로더 초기화 완료 - 100% 검증된 인프라 사용")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        A2A 프로토콜에 따른 통합 데이터 로딩 실행
        
        5단계 워크플로우:
        1. 의도 분석 (LLM First)
        2. 파일 발견 및 선택
        3. 로딩 전략 수립
        4. 데이터 로딩 및 검증
        5. 공유 저장소 저장
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        # TaskUpdater 초기화 (A2A SDK 표준 패턴)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # A2A 태스크 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 요청 추출
            user_query = self._extract_user_query(context)
            if not user_query:
                await task_updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message("❌ 유효한 사용자 요청을 찾을 수 없습니다.")
                )
                return
            
            logger.info(f"📝 사용자 요청: {user_query}")
            
            # === 1단계: LLM First 의도 분석 ===
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧠 LLM 기반 의도 분석 중...")
            )
            
            a2a_context = A2AContext(context)
            
            intent = await self.llm_engine.analyze_intent(user_query, a2a_context)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"✅ 의도 분석 완료: {intent.intent_type.value} (신뢰도: {intent.confidence:.2f})")
            )
            
            # === 2단계: 파일 발견 및 선택 ===
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("📁 사용 가능한 데이터 파일 스캔 중...")
            )
            
            available_files = await self._discover_available_files()
            
            if not available_files:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("""📁 사용 가능한 데이터 파일이 없습니다.

### 💡 해결 방법:
1. **데이터 파일 업로드**: CSV, Excel, JSON 등 파일을 업로드해주세요
2. **지원 형식**: .csv, .xlsx, .xls, .json, .parquet, .feather, .txt, .tsv
3. **권장 위치**: `a2a_ds_servers/artifacts/data/` 폴더

### 🔧 다음 단계:
데이터 파일을 준비한 후 다시 요청해주세요.""")
                )
                return
            
            selected_file = await self.llm_engine.select_optimal_file(intent, available_files)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"🎯 파일 선택 완료: {Path(selected_file).name} ({len(available_files)}개 중 선택)")
            )
            
            # === 3단계: 로딩 전략 수립 ===
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("⚡ 최적 로딩 전략 수립 중...")
            )
            
            loading_strategy = await self.llm_engine.create_loading_strategy(selected_file, intent)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"📋 로딩 전략: {loading_strategy.encoding} 인코딩, 캐시 {'활성화' if loading_strategy.use_cache else '비활성화'}")
            )
            
            # === 4단계: 데이터 로딩 및 검증 ===
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("📊 데이터 로딩 및 품질 검증 중...")
            )
            
            smart_df = await self.file_connector.load_file(selected_file, loading_strategy, a2a_context)
            
            # 빈 데이터 체크
            if smart_df.is_empty():
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(f"""⚠️ 로딩된 데이터가 비어있습니다: {Path(selected_file).name}

### 🔧 해결 방법:
1. **파일 내용 확인**: 데이터가 올바르게 포함되어 있는지 확인
2. **인코딩 문제**: UTF-8 또는 CP949 인코딩으로 저장 시도
3. **형식 확인**: CSV의 경우 헤더와 구분자 확인

### 💡 추천:
Data Cleaning Agent를 사용하여 데이터 품질을 개선할 수 있습니다.""")
                )
                return
            
            # 자동 프로파일링
            profile = await smart_df.auto_profile()
            quality_report = await smart_df.validate_quality()
            
            # === 5단계: 공유 저장소 저장 ===
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("💾 공유 저장소에 저장 중...")
            )
            
            saved_info = await self._save_to_shared_storage(smart_df, context.task_id)
            
            # 성공 통계 업데이트
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_loads"] += 1
            self.stats["average_load_time"] = (
                (self.stats["average_load_time"] * (self.stats["successful_loads"] - 1) + processing_time) 
                / self.stats["successful_loads"]
            )
            
            # 최종 성공 응답
            success_message = self._generate_success_response(
                smart_df, profile, quality_report, saved_info, processing_time
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(success_message)
            )
            
            logger.info(f"✅ 데이터 로딩 성공: {smart_df.shape} in {processing_time:.2f}s")
            
        except Exception as e:
            self.stats["failed_loads"] += 1
            error_message = f"❌ 데이터 로딩 실패: {str(e)}"
            
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(error_message)
            )
            
            logger.error(f"데이터 로딩 오류: {e}", exc_info=True)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """에이전트 실행 취소"""
        try:
            task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await task_updater.update_status(
                TaskState.cancelled,
                message=task_updater.new_agent_message(parts=[TextPart(text="❌ 데이터 로딩 작업이 취소되었습니다.")])
            )
            logger.info("데이터 로딩 작업 취소됨")
        except Exception as e:
            logger.error(f"작업 취소 중 오류: {e}")
    
    def _extract_user_query(self, context: RequestContext) -> str:
        """A2A 컨텍스트에서 사용자 쿼리 추출 (표준 A2A 패턴)"""
        try:
            # A2A 표준: context.message.parts에서 텍스트 추출
            if hasattr(context, 'message') and context.message and hasattr(context.message, 'parts'):
                parts = []
                for part in context.message.parts:
                    # A2A SDK 표준 구조: part.root.text 또는 part.root.kind == "text"
                    if hasattr(part, 'root'):
                        if hasattr(part.root, 'text'):
                            parts.append(part.root.text)
                        elif hasattr(part.root, 'kind') and part.root.kind == "text":
                            if hasattr(part.root, 'content'):
                                parts.append(part.root.content)
                    # 직접 text 속성이 있는 경우
                    elif hasattr(part, 'text'):
                        parts.append(part.text)
                
                user_query = " ".join(parts).strip()
                if user_query:
                    return user_query
            
            # 폴백 1: context.get_user_input() 메서드 사용
            if hasattr(context, 'get_user_input'):
                user_input = context.get_user_input()
                if user_input:
                    return str(user_input).strip()
                    
            # 폴백 2: 기본 메시지
            return "데이터 파일을 로드해주세요"
            
        except Exception as e:
            logger.warning(f"사용자 쿼리 추출 실패: {e}")
            return "데이터 파일을 로드해주세요"
    
    async def _discover_available_files(self) -> List[str]:
        """사용 가능한 데이터 파일 발견"""
        try:
            from unified_data_system.utils.file_scanner import FileScanner
            
            scanner = FileScanner()
            
            # FileScanner 내장 스캔 기능 사용 (절대 경로 적용됨)
            all_files = await scanner.scan_data_files()
            
            # 추가적인 현재 디렉토리 스캔 (절대 경로로)
            current_dir = Path.cwd()
            additional_paths = [
                current_dir / "a2a_ds_servers" / "artifacts" / "data",
                current_dir / "data",
                current_dir / "datasets", 
                current_dir / "files",
                current_dir / "uploads",
                current_dir  # 현재 디렉토리
            ]
            
            for path in additional_paths:
                if path.exists() and path.is_dir():
                    try:
                        extra_files = await scanner.scan_directory(str(path))
                        all_files.extend(extra_files)
                    except Exception as e:
                        logger.debug(f"경로 스캔 실패 ({path}): {e}")
            
            # 중복 제거 및 정렬  
            unique_files = list(set(all_files))
            unique_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # 최신 파일 우선
            
            logger.info(f"📁 발견된 파일: {len(unique_files)}개")
            for file in unique_files[:5]:  # 첫 5개 파일 로그
                logger.info(f"  - {file}")
            
            return unique_files
            
        except Exception as e:
            logger.error(f"파일 발견 오류: {e}")
            return []
    
    async def _save_to_shared_storage(self, smart_df: SmartDataFrame, task_id: str) -> Dict[str, Any]:
        """공유 저장소에 데이터 저장"""
        try:
            # 고유한 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loaded_data_{task_id}_{timestamp}.csv"
            filepath = self.shared_data_path / filename
            
            # CSV로 저장
            smart_df.to_csv(filepath, index=False)
            
            # 메타데이터 저장
            metadata_file = filepath.with_suffix('.json')
            metadata = {
                "task_id": task_id,
                "filename": filename,
                "filepath": str(filepath),
                "shape": smart_df.shape,
                "columns": list(smart_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in smart_df.dtypes.items()},
                "created_at": datetime.now().isoformat(),
                "source_metadata": smart_df.metadata
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 데이터 저장 완료: {filepath}")
            
            return {
                "filepath": str(filepath),
                "filename": filename,
                "metadata_file": str(metadata_file),
                "size_mb": filepath.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"공유 저장소 저장 실패: {e}")
            raise
    
    def _generate_success_response(self, 
                                  smart_df: SmartDataFrame, 
                                  profile: DataProfile, 
                                  quality_report: QualityReport,
                                  saved_info: Dict[str, Any],
                                  processing_time: float) -> str:
        """성공 응답 메시지 생성"""
        
        quality_emoji = "🟢" if quality_report.overall_score >= 0.8 else "🟡" if quality_report.overall_score >= 0.6 else "🔴"
        
        response = f"""✅ **데이터 로딩 성공!**

### 📊 **로딩된 데이터 정보**
- **형태**: {smart_df.shape[0]:,}행 × {smart_df.shape[1]:,}열
- **파일명**: `{saved_info['filename']}`
- **크기**: {saved_info['size_mb']:.2f} MB
- **처리 시간**: {processing_time:.2f}초

### {quality_emoji} **데이터 품질 분석**
- **전체 품질 점수**: {quality_report.overall_score:.1%}
- **완전성**: {quality_report.completeness:.1%} | **일관성**: {quality_report.consistency:.1%}
- **유효성**: {quality_report.validity:.1%} | **정확성**: {quality_report.accuracy:.1%}

### 📋 **컬럼 정보**
{', '.join([f"`{col}` ({str(dtype)})" for col, dtype in smart_df.dtypes.items()])}

### 📈 **샘플 데이터**
```
{smart_df.head(3).to_string()}
```"""

        # 품질 이슈가 있는 경우 추가 정보
        if quality_report.issues:
            response += f"\n\n### ⚠️ **품질 이슈**\n"
            for issue in quality_report.issues[:3]:  # 최대 3개까지
                response += f"- {issue}\n"
        
        # 권장사항
        if quality_report.recommendations:
            response += f"\n### 💡 **권장사항**\n"
            for rec in quality_report.recommendations[:2]:  # 최대 2개까지
                response += f"- {rec}\n"
        
        response += f"""

### 🚀 **다음 단계**
이제 로딩된 데이터를 다른 CherryAI 에이전트들과 함께 사용할 수 있습니다:
- **Data Cleaning Agent** (8306): 데이터 정리 및 전처리
- **EDA Tools Agent** (8312): 탐색적 데이터 분석  
- **Data Visualization Agent** (8308): 시각화 생성
- **Feature Engineering Agent** (8310): 특성 엔지니어링

### 📍 **저장 위치**
`{saved_info['filepath']}`"""

        return response


def create_agent_card() -> AgentCard:
    """A2A 에이전트 카드 생성"""
    
    skill = AgentSkill(
        id="unified_data_loading",
        name="Unified Data Loading & Processing",
        description="pandas_agent 패턴 기반 통합 데이터 로딩 시스템. LLM First 원칙으로 지능형 파일 선택, UTF-8 문제 해결, 자동 품질 검증 제공",
        tags=[
            "data-loading", "file-processing", "quality-validation", 
            "encoding-detection", "llm-first", "pandas-agent-pattern",
            "utf8-support", "intelligent-selection", "caching"
        ],
        examples=[
            "사용 가능한 데이터 파일을 로드해주세요",
            "CSV 파일을 분석용으로 준비해주세요", 
            "데이터 품질을 검증하고 로딩해주세요",
            "Excel 파일을 DataFrame으로 변환해주세요",
            "가장 적합한 데이터 파일을 자동으로 선택해주세요"
        ]
    )
    
    return AgentCard(
        name="CherryAI Unified Data Loader",
        description="pandas_agent 패턴 기반 통합 데이터 로딩 시스템. 12개 A2A 에이전트의 표준 데이터 공급원으로 LLM First 원칙과 지능형 파일 처리를 통해 완벽한 데이터 로딩 경험을 제공합니다.",
        url="http://localhost:8307/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )


def main():
    """통합 데이터 로더 서버 실행"""
    
    # 환경 설정
    logging.basicConfig(level=logging.INFO)
    
    # A2A 서버 구성 요소
    agent_card = create_agent_card()
    
    request_handler = DefaultRequestHandler(
        agent_executor=UnifiedDataLoaderExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🍒 CherryAI 통합 데이터 로더 서버 시작")
    print("📊 pandas_agent 패턴 기반 표준 데이터 로딩 시스템")
    print(f"🌐 서버 URL: http://localhost:8307")
    print(f"📋 에이전트 카드: http://localhost:8307/.well-known/agent.json")
    print("")
    print("✨ 주요 특징:")
    print("  - 🧠 LLM First 원칙: 지능형 의도 분석 및 파일 선택")
    print("  - 🔍 UTF-8 문제 완전 해결: 다중 인코딩 자동 감지")
    print("  - 📊 자동 품질 검증: SmartDataFrame + 실시간 프로파일링")
    print("  - ⚡ 고성능 캐싱: LRU + TTL + 태그 기반 캐시 시스템")
    print("  - 🔗 다중 형식 지원: CSV, Excel, JSON, Parquet 등")
    print("  - 🎯 A2A 표준: SDK 0.2.9 완벽 준수")
    print("  - 🍒 CherryAI 표준: 12개 에이전트 통합 데이터 공급원")
    print("")
    print("🚀 서버가 시작됩니다...")
    
    # 서버 실행
    uvicorn.run(
        server.build(), 
        host="0.0.0.0", 
        port=8307, 
        log_level="info"
    )


if __name__ == "__main__":
    main() 