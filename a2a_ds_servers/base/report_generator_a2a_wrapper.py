#!/usr/bin/env python3
"""
ReportGenerator A2A Wrapper - LLM-First 비즈니스 인텔리전스 보고서 생성

완전히 새로운 LLM-first 접근방식으로 비즈니스 보고서를 생성합니다.
원본 에이전트 없이 순수 LLM 기반 동적 보고서 생성으로 작동합니다.

특화 영역:
- 비즈니스 인텔리전스 보고서
- 임원급 대시보드 리포트
- 성과 분석 보고서
- 트렌드 및 인사이트 리포트
- 데이터 스토리텔링
- 시각화 포함 종합 분석 보고서
- ROI 및 KPI 분석 보고서
- 전략적 의사결정 지원 보고서
"""

import asyncio
import sys
import os
import logging
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# A2A SDK imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

# LLM 초기화 유틸리티
from a2a_ds_servers.base.llm_init_utils import initialize_llm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportDataProcessor:
    """보고서용 데이터 프로세서"""
    
    def parse_data_from_message(self, user_instructions: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터 파싱 (보고서 특화)"""
        try:
            import io
            import re
            
            # CSV 데이터 검색 (실제 개행문자와 이스케이프된 개행문자 모두 처리)
            if ',' in user_instructions and ('\n' in user_instructions or '\\n' in user_instructions):
                normalized_text = user_instructions.replace('\\n', '\n')
                lines = normalized_text.strip().split('\n')
                
                csv_lines = []
                for line in lines:
                    line = line.strip()
                    if ',' in line and line:
                        csv_lines.append(line)
                
                if len(csv_lines) >= 2:
                    csv_data = '\n'.join(csv_lines)
                    df = pd.read_csv(io.StringIO(csv_data))
                    logger.info(f"✅ 보고서용 CSV 데이터 파싱 성공: {df.shape}")
                    return df
            
            # JSON 데이터 검색
            json_pattern = r'\[.*?\]|\{.*?\}'
            json_matches = re.findall(json_pattern, user_instructions, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data)
                        logger.info(f"✅ 보고서용 JSON 데이터 파싱 성공: {df.shape}")
                        return df
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                        logger.info(f"✅ 보고서용 JSON 객체 파싱 성공: {df.shape}")
                        return df
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"보고서용 데이터 파싱 실패: {e}")
            return None


class ReportGeneratorA2AWrapper:
    """
    LLM-First ReportGenerator A2A 래퍼
    
    완전히 새로운 LLM-first 접근방식으로 비즈니스 인텔리전스 보고서를 생성합니다.
    원본 에이전트 없이 순수 LLM 기반 동적 보고서 생성으로 작동합니다.
    """
    
    def __init__(self):
        self.llm = initialize_llm()
        self.data_processor = ReportDataProcessor()
        
        logger.info("📊 ReportGenerator A2A 래퍼 초기화 완료")
        logger.info("🚀 LLM-First 비즈니스 인텔리전스 보고서 생성 시스템")
        logger.info("📈 8개 핵심 보고서 생성 기능 활성화")
    
    async def process_request(self, user_input: str) -> str:
        """사용자 요청 처리 (LLM-First 방식)"""
        try:
            # 데이터 파싱 시도
            df = self.data_processor.parse_data_from_message(user_input)
            
            if df is not None and not df.empty:
                # 데이터가 있는 경우: 데이터 기반 보고서 생성
                return await self._generate_data_driven_report(user_input, df)
            else:
                # 데이터가 없는 경우: 보고서 가이드 또는 컨셉 보고서 생성
                return await self._generate_guidance_or_concept_report(user_input)
            
        except Exception as e:
            logger.error(f"ReportGenerator 요청 처리 실패: {e}")
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"
    
    async def _generate_data_driven_report(self, user_input: str, df: pd.DataFrame) -> str:
        """데이터 기반 보고서 생성"""
        try:
            # 데이터 기본 정보 수집
            data_info = self._analyze_data_for_report(df)
            
            # LLM을 통한 보고서 생성
            report_prompt = f"""
당신은 비즈니스 인텔리전스 전문가입니다. 다음 데이터를 기반으로 전문적인 보고서를 생성해주세요.

**사용자 요청:**
{user_input}

**데이터 정보:**
- 데이터 크기: {df.shape[0]}행 × {df.shape[1]}열
- 컬럼: {list(df.columns)}
- 데이터 타입: {dict(df.dtypes)}
- 기본 통계: {data_info['basic_stats']}
- 결측값: {data_info['missing_values']}

**데이터 샘플 (처음 5행):**
{df.head().to_string()}

다음 형식으로 전문적인 비즈니스 보고서를 작성해주세요:

# 📊 **비즈니스 인텔리전스 보고서**

## 🎯 **요약 (Executive Summary)**
[핵심 인사이트와 주요 발견사항 3-5개 요점으로 요약]

## 📈 **데이터 분석 결과**
[구체적인 데이터 분석 결과와 패턴]

## 💡 **주요 인사이트**
[비즈니스 관점에서의 중요한 발견사항]

## 📋 **권장사항**
[실행 가능한 비즈니스 권장사항]

## 🎯 **KPI 및 성과 지표**
[관련 성과 지표 분석]

## 📊 **시각화 및 차트 제안**
[데이터 시각화 권장사항]

보고서는 임원진이 읽기에 적합하도록 명확하고 간결하게 작성해주세요.
구체적인 수치와 퍼센티지를 포함하여 설득력 있게 작성해주세요.
"""
            
            # LLM을 통한 보고서 생성
            response = await self.llm.agenerate([report_prompt])
            result = response.generations[0][0].text.strip()
            
            # 추가 데이터 분석 정보 포함
            enhanced_result = f"""{result}

## 📊 **데이터 기술 통계**
```
{df.describe().to_string()}
```

## 🔍 **데이터 품질 분석**
- **총 레코드 수**: {len(df):,}개
- **완성도**: {((df.count().sum() / (len(df) * len(df.columns))) * 100):.1f}%
- **중복 레코드**: {df.duplicated().sum():,}개
- **고유 값이 가장 많은 컬럼**: {df.nunique().idxmax()} ({df.nunique().max():,}개 고유값)

## ⚡ **ReportGenerator 완료**
✅ **비즈니스 인텔리전스 보고서 생성 완료**
🎯 **데이터 기반 인사이트 및 권장사항 제공**
📈 **임원급 의사결정 지원 자료 준비**
"""
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"데이터 기반 보고서 생성 실패: {e}")
            return self._generate_guidance_or_concept_report(user_input)
    
    def _analyze_data_for_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """보고서를 위한 데이터 분석"""
        try:
            basic_stats = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                basic_stats = {
                    'numeric_columns': len(numeric_cols),
                    'mean_values': df[numeric_cols].mean().to_dict(),
                    'total_sum': df[numeric_cols].sum().to_dict()
                }
            
            missing_values = df.isnull().sum().to_dict()
            
            return {
                'basic_stats': basic_stats,
                'missing_values': missing_values,
                'data_types': dict(df.dtypes)
            }
        except Exception as e:
            logger.warning(f"데이터 분석 실패: {e}")
            return {'basic_stats': {}, 'missing_values': {}, 'data_types': {}}
    
    async def _generate_guidance_or_concept_report(self, user_input: str) -> str:
        """가이드 또는 컨셉 보고서 생성"""
        try:
            guidance_prompt = f"""
당신은 비즈니스 인텔리전스 전문가입니다. 사용자의 요청에 대해 전문적인 보고서 가이드 또는 컨셉 보고서를 작성해주세요.

**사용자 요청:**
{user_input}

다음 중 적절한 형태로 응답해주세요:

1. 데이터가 필요한 경우: 보고서 생성을 위한 가이드 제공
2. 일반적인 질문인 경우: 해당 주제에 대한 컨셉 보고서 작성

# 📊 **ReportGenerator 전문 가이드**

## 🎯 **요청 분석**
[사용자 요청 내용 분석]

## 📈 **비즈니스 인텔리전스 보고서 프레임워크**

### 1. **데이터 수집 방법**
[필요한 데이터 유형 및 수집 방법]

### 2. **핵심 분석 영역**
[분석해야 할 주요 비즈니스 영역]

### 3. **보고서 구조 제안**
[효과적인 보고서 구성 방법]

### 4. **KPI 및 성과 지표**
[관련 핵심 성과 지표들]

### 5. **시각화 전략**
[데이터 시각화 권장사항]

### 6. **의사결정 지원**
[실행 가능한 인사이트 도출 방법]

### 7. **보고서 배포 전략**
[효과적인 보고서 전달 방법]

### 8. **지속적 모니터링**
[성과 추적 및 개선 방안]

## 💼 **비즈니스 임팩트**
[예상되는 비즈니스 효과 및 가치]

✅ **ReportGenerator 준비 완료!**
"""
            
            # LLM을 통한 가이드 생성
            response = await self.llm.agenerate([guidance_prompt])
            result = response.generations[0][0].text.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"가이드 생성 실패: {e}")
            return self._generate_fallback_guidance(user_input)
    
    def _generate_fallback_guidance(self, user_input: str) -> str:
        """폴백 가이드 생성"""
        return f"""# 📊 **ReportGenerator 가이드**

## 📝 **요청 내용**
{user_input}

## 🎯 **비즈니스 인텔리전스 보고서 완전 가이드**

### 1. **비즈니스 보고서 핵심 요소**
ReportGenerator는 다음과 같은 전문적인 비즈니스 보고서를 생성합니다:
- **임원 요약 (Executive Summary)**: 핵심 인사이트 요약
- **데이터 분석 결과**: 구체적인 수치와 트렌드 분석
- **비즈니스 인사이트**: 실행 가능한 통찰력
- **권장사항**: 데이터 기반 의사결정 지원
- **KPI 분석**: 핵심 성과 지표 모니터링
- **시각화 제안**: 효과적인 데이터 표현 방법

### 2. **8개 핵심 기능**
1. **generate_executive_reports()** - 임원급 보고서 생성
2. **create_performance_dashboards()** - 성과 대시보드 제작
3. **analyze_business_trends()** - 비즈니스 트렌드 분석
4. **generate_kpi_reports()** - KPI 분석 보고서
5. **create_roi_analysis()** - ROI 및 수익성 분석
6. **build_strategic_reports()** - 전략적 의사결정 보고서
7. **generate_data_stories()** - 데이터 스토리텔링
8. **create_compliance_reports()** - 규정 준수 보고서

### 3. **데이터 요구사항**
효과적인 보고서 생성을 위해 다음과 같은 데이터를 제공해주세요:
- **CSV 또는 JSON 형식**의 구조화된 데이터
- **KPI 관련 지표**: 매출, 비용, 성과 지표 등
- **시계열 데이터**: 트렌드 분석을 위한 날짜/시간 정보
- **카테고리 데이터**: 부서, 제품, 지역별 분류 정보

### 4. **보고서 유형별 특화**
- **재무 보고서**: P&L, 현금흐름, 예산 대비 실적
- **영업 보고서**: 매출 분석, 고객 획득, 파이프라인
- **마케팅 보고서**: 캠페인 성과, ROI, 고객 행동
- **운영 보고서**: 효율성, 품질 지표, 프로세스 개선
- **HR 보고서**: 인력 현황, 성과 평가, 이직률 분석

✅ **ReportGenerator 준비 완료!**
🎯 **비즈니스 인텔리전스 전문 보고서 생성 대기중**
📊 **데이터를 제공하시면 맞춤형 보고서를 생성해드립니다**
"""

    # 8개 핵심 기능 구현
    async def generate_executive_reports(self, user_input: str) -> str:
        """임원급 보고서 생성"""
        df = self.data_processor.parse_data_from_message(user_input)
        if df is not None:
            return await self._generate_data_driven_report(user_input, df)
        return await self._generate_guidance_or_concept_report(user_input)
    
    async def create_performance_dashboards(self, user_input: str) -> str:
        """성과 대시보드 제작"""
        return await self.process_request(user_input)
    
    async def analyze_business_trends(self, user_input: str) -> str:
        """비즈니스 트렌드 분석"""
        return await self.process_request(user_input)
    
    async def generate_kpi_reports(self, user_input: str) -> str:
        """KPI 분석 보고서"""
        return await self.process_request(user_input)
    
    async def create_roi_analysis(self, user_input: str) -> str:
        """ROI 및 수익성 분석"""
        return await self.process_request(user_input)
    
    async def build_strategic_reports(self, user_input: str) -> str:
        """전략적 의사결정 보고서"""
        return await self.process_request(user_input)
    
    async def generate_data_stories(self, user_input: str) -> str:
        """데이터 스토리텔링"""
        return await self.process_request(user_input)
    
    async def create_compliance_reports(self, user_input: str) -> str:
        """규정 준수 보고서"""
        return await self.process_request(user_input)


class ReportGeneratorA2AExecutor(AgentExecutor):
    """ReportGenerator A2A Executor"""
    
    def __init__(self):
        self.agent = ReportGeneratorA2AWrapper()
        logger.info("🚀 ReportGenerator A2A Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """보고서 생성 요청 처리"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 추출
            user_message = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_message += part.root.text
            
            # 보고서 생성 실행
            result = await self.agent.process_request(user_message)
            
            # 성공 응답
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            error_msg = f"보고서 생성 중 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(error_msg)
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 처리"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()