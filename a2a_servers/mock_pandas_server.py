#!/usr/bin/env python3
"""
Mock A2A Pandas Server - UI 테스트용
실제 A2A 프로토콜과 호환되는 간단한 구현
"""

import pandas as pd
import os
import sys
import uvicorn
import logging
from typing import Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CherryAI imports
from core.data_manager import DataManager

# 전역 데이터 매니저
data_manager = DataManager()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(title="Mock Pandas A2A Server", version="1.0.0")

# Pydantic 모델
class MessagePart(BaseModel):
    text: str
    kind: str = "text"

class Message(BaseModel):
    messageId: str
    role: str
    parts: list[MessagePart]

class MessageParams(BaseModel):
    message: Message

class A2ARequest(BaseModel):
    id: str
    method: str = "message/send"  # A2A 표준 메서드
    params: MessageParams

class A2AResult(BaseModel):
    messageId: str
    role: str = "assistant" 
    parts: list[MessagePart]

class A2AResponse(BaseModel):
    id: str
    result: A2AResult

def perform_analysis(df: pd.DataFrame, df_id: str, prompt: str) -> str:
    """실제 데이터 분석 수행"""
    import numpy as np
    
    # 기본 정보
    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 데이터 품질
    missing_data = df.isnull().sum()
    completeness = ((total_rows * total_cols - missing_data.sum()) / (total_rows * total_cols)) * 100
    
    # 타임스탬프
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 기본 통계
    stats_table = ""
    if numeric_cols:
        stats_table = df.describe().round(2).to_markdown()
    else:
        stats_table = "숫자형 데이터가 없습니다."
    
    # 결측값 정보
    missing_info = ""
    if missing_data.sum() > 0:
        missing_info = "\n".join([f"- **{col}**: {count}개 ({count/total_rows*100:.1f}%)" 
                                  for col, count in missing_data.items() if count > 0])
    else:
        missing_info = "✅ 결측값이 없습니다."
    
    # 범주형 변수 분포
    categorical_info = ""
    for col in categorical_cols[:3]:
        value_counts = df[col].value_counts().head(5)
        categorical_info += f"\n**{col}**:\n"
        for value, count in value_counts.items():
            categorical_info += f"- {value}: {count}개 ({count/total_rows*100:.1f}%)\n"
    
    # 상관관계 분석
    correlation_info = ""
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr.append(f"- {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        if high_corr:
            correlation_info = "### 🔗 주요 상관관계 (|r| > 0.5)\n" + "\n".join(high_corr)
        else:
            correlation_info = "### 🔗 상관관계\n강한 상관관계(|r| > 0.5)를 보이는 변수 쌍이 없습니다."
    
    # 생존율 분석 (Titanic 데이터의 경우)
    survival_analysis = ""
    if 'Survived' in df.columns:
        survival_rate = df['Survived'].mean() * 100
        survival_analysis = f"""
### ⚓ 생존율 분석

**전체 생존율**: {survival_rate:.1f}%

**성별별 생존율**:
"""
        if 'Sex' in df.columns:
            sex_survival = df.groupby('Sex')['Survived'].mean() * 100
            for sex, rate in sex_survival.items():
                survival_analysis += f"- {sex}: {rate:.1f}%\n"
        
        if 'Pclass' in df.columns:
            survival_analysis += "\n**객실 등급별 생존율**:\n"
            class_survival = df.groupby('Pclass')['Survived'].mean() * 100
            for pclass, rate in class_survival.items():
                survival_analysis += f"- {pclass}등석: {rate:.1f}%\n"
    
    # 최종 보고서 구성
    final_result = f"""# 📊 **완전한 EDA 분석 보고서**

**분석 대상**: `{df_id}`  
**분석 일시**: {timestamp}  
**요청**: {prompt}

---

## 📋 **데이터 개요**

| 항목 | 값 |
|------|-----|
| 📏 데이터 크기 | **{total_rows:,}** 행 × **{total_cols}** 열 |
| ✅ 완성도 | **{completeness:.1f}%** |
| 🔢 숫자형 변수 | **{len(numeric_cols)}개** |
| 📝 범주형 변수 | **{len(categorical_cols)}개** |
| 💾 메모리 사용량 | **{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB** |

---

## 🔍 **기본 통계**

{stats_table}

---

## ❌ **결측값 현황**

{missing_info}

---

## 📊 **범주형 변수 분포**

{categorical_info}

---

{correlation_info}

{survival_analysis}

---

## 💡 **핵심 인사이트**

1. **📏 데이터 규모**: {total_rows:,}개 관측값으로 {"**충분한**" if total_rows > 1000 else "**제한적인**"} 분석이 가능합니다.

2. **✅ 데이터 품질**: {completeness:.1f}%의 완성도로 {"**우수한**" if completeness > 95 else "**보통**" if completeness > 80 else "**개선이 필요한**"} 수준입니다.

3. **🔢 변수 구성**: {len(numeric_cols)}개의 숫자형 변수와 {len(categorical_cols)}개의 범주형 변수로 **다양한 관점의 분석**이 가능합니다.

---

## 📋 **추천 분석 방향**

🎯 **다음 단계로 진행할 수 있는 분석**:

1. **📈 시각화 분석**: 히스토그램, 박스플롯으로 분포 확인
2. **🔗 상관관계 히트맵**: 변수 간 관계의 시각적 표현  
3. **🎯 타겟 분석**: 특정 결과 변수에 영향을 미치는 요인 탐색
4. **🧹 데이터 전처리**: 결측값 처리 및 이상값 제거
5. **🤖 머신러닝**: 예측 모델 구축 가능성 검토

---

✅ **분석 완료**  
🕐 **처리 시간**: < 2초  
🔧 **분석 엔진**: Mock Pandas A2A Server
"""
    
    return final_result

@app.get("/.well-known/agent.json")
async def get_agent_card():
    """A2A 표준 Agent Card"""
    return {
        "name": "Pandas Data Analyst (Mock)",
        "description": "Mock A2A server for testing - Expert data analyst using pandas",
        "version": "1.0.0-mock",
        "url": "http://localhost:10001",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "analyze_data",
                "name": "Data Analysis",
                "description": "Analyze datasets using pandas and provide comprehensive insights",
                "tags": ["data", "analysis", "pandas", "statistics", "eda"],
                "examples": ["analyze the titanic dataset", "perform EDA on sales data"]
            }
        ]
    }

@app.post("/")
async def handle_a2a_request(request: A2ARequest) -> A2AResponse:
    """A2A 메시지 처리 - 표준 message/send 프로토콜"""
    logger.info(f"🎯 A2A Request received: {request.method}")
    
    try:
        # 메시지에서 텍스트 추출
        user_text = ""
        if request.params.message.parts:
            for part in request.params.message.parts:
                user_text += part.text + " "
        user_text = user_text.strip()
        
        logger.info(f"📝 User request: {user_text}")
        
        # 사용 가능한 데이터프레임 확인
        available_dfs = data_manager.list_dataframes()
        logger.info(f"💾 Available dataframes: {available_dfs}")
        
        if not available_dfs:
            analysis_result = """❌ **데이터 없음**

**문제**: 아직 업로드된 데이터셋이 없습니다.

**해결방법:**
1. 🔄 **데이터 로더** 페이지로 이동
2. 📁 CSV, Excel 등의 데이터 파일 업로드  
3. 📊 다시 돌아와서 데이터 분석 요청

**현재 사용 가능한 데이터셋**: 없음
"""
        else:
            # 첫 번째 데이터프레임 사용
            df_id = available_dfs[0]
            df = data_manager.get_dataframe(df_id)
            
            if df is None:
                analysis_result = f"❌ 데이터셋 '{df_id}'를 로드할 수 없습니다."
            else:
                # 실제 분석 수행
                analysis_result = perform_analysis(df, df_id, user_text)
        
        # A2A 응답 생성
        response = A2AResponse(
            id=request.id,
            result=A2AResult(
                messageId=f"response-{request.id}",
                role="assistant",
                parts=[MessagePart(text=analysis_result, kind="text")]
            )
        )
        
        logger.info("✅ Analysis completed and response generated")
        return response
        
    except Exception as e:
        logger.error(f"💥 Analysis failed: {e}", exc_info=True)
        
        error_response = A2AResponse(
            id=request.id,
            result=A2AResult(
                messageId=f"error-{request.id}",
                role="assistant", 
                parts=[MessagePart(text=f"❌ 분석 실패: {str(e)}", kind="text")]
            )
        )
        return error_response

if __name__ == "__main__":
    logger.info("🚀 Starting Mock Pandas A2A Server...")
    logger.info("🎯 이 서버는 UI 테스트를 위한 Mock 구현입니다")
    logger.info("📊 실제 데이터 분석 기능을 제공합니다")
    
    uvicorn.run(app, host="0.0.0.0", port=10001) 