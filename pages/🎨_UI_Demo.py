#!/usr/bin/env python3
"""
CherryAI UI/UX 개선 데모 페이지
새로운 사용자 친화적 UI 컴포넌트들을 시연하고 테스트하는 페이지
"""

import streamlit as st
import time
from datetime import datetime
import json

# Python 경로 설정
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# UI 컴포넌트 임포트
from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults
from ui.message_translator import MessageRenderer
from ui.sidebar_components import render_sidebar

def main():
    """메인 데모 페이지"""
    st.set_page_config(
        page_title="UI/UX Demo", 
        layout="wide", 
        page_icon="🎨",
        initial_sidebar_state="expanded"
    )
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 페이지 헤더
    st.title("🎨 CherryAI UI/UX 개선 데모")
    st.markdown("""
    이 페이지에서는 새롭게 개선된 사용자 친화적 UI 컴포넌트들을 시연합니다.
    기존의 기술적이고 복잡한 인터페이스를 아름답고 직관적인 경험으로 변환했습니다.
    """)
    
    # 탭으로 구분된 데모 섹션
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 사고 과정 스트리밍", 
        "📋 계획 시각화", 
        "🔄 메시지 번역", 
        "✨ 결과 표시"
    ])
    
    with tab1:
        demo_thinking_stream()
    
    with tab2:
        demo_plan_visualization()
    
    with tab3:
        demo_message_translation()
    
    with tab4:
        demo_beautiful_results()

def demo_thinking_stream():
    """사고 과정 스트리밍 데모"""
    st.header("🧠 AI 사고 과정 실시간 스트리밍")
    
    st.markdown("""
    ### 개선 전 vs 개선 후
    
    **개선 전:** 단순한 "Loading..." 또는 기술적 상태 메시지
    **개선 후:** AI의 사고 과정을 실시간으로 시각화하여 사용자가 무엇이 일어나고 있는지 이해할 수 있음
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 사고 과정 시작", key="thinking_demo"):
            demo_thinking_process()
    
    with col2:
        if st.button("⚡ 빠른 사고 과정", key="quick_thinking"):
            demo_quick_thinking()

def demo_thinking_process():
    """상세한 사고 과정 데모"""
    thinking_container = st.container()
    thinking = ThinkingStream(thinking_container)
    
    # 사고 과정 시뮬레이션
    steps = [
        ("데이터 분석 요청을 받았습니다...", "analysis", 1.5),
        ("데이터의 구조와 품질을 검토하고 있습니다.", "data_processing", 2.0),
        ("적절한 분석 방법을 선택하고 있습니다.", "analysis", 1.8),
        ("시각화 전략을 수립하고 있습니다.", "visualization", 1.5),
        ("통계적 검증 방법을 결정하고 있습니다.", "analysis", 1.2),
        ("최종 보고서 구조를 계획하고 있습니다.", "planning", 1.0)
    ]
    
    thinking.start_thinking("복잡한 데이터 분석 작업을 시작합니다...")
    
    for thought, thought_type, delay in steps:
        time.sleep(delay)
        thinking.add_thought(thought, thought_type)
    
    thinking.finish_thinking("완벽한 분석 계획이 수립되었습니다! 🎉")

def demo_quick_thinking():
    """빠른 사고 과정 데모"""
    thinking_container = st.container()
    thinking = ThinkingStream(thinking_container)
    
    thinking.start_thinking("간단한 분석을 시작합니다...")
    time.sleep(0.5)
    thinking.add_thought("데이터 로딩 완료", "success")
    time.sleep(0.5)
    thinking.add_thought("기본 통계 계산 중", "data_processing")
    time.sleep(0.5)
    thinking.finish_thinking("분석 완료!")

def demo_plan_visualization():
    """계획 시각화 데모"""
    st.header("📋 아름다운 계획 시각화")
    
    st.markdown("""
    ### 개선 전 vs 개선 후
    
    **개선 전:** 단순한 텍스트 목록으로 계획 표시
    **개선 후:** 시각적 카드와 애니메이션으로 각 단계를 명확하게 표현
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 데이터 분석 계획", key="data_plan"):
            demo_data_analysis_plan()
    
    with col2:
        if st.button("🤖 ML 파이프라인 계획", key="ml_plan"):
            demo_ml_pipeline_plan()

def demo_data_analysis_plan():
    """데이터 분석 계획 시각화"""
    plan_viz = PlanVisualization()
    
    sample_plan = [
        {"agent_name": "Data Validator", "skill_name": "데이터 품질 검증 및 정제"},
        {"agent_name": "EDA Analyst", "skill_name": "탐색적 데이터 분석 수행"},
        {"agent_name": "Statistical Analyst", "skill_name": "통계적 가설 검정"},
        {"agent_name": "Visualization Expert", "skill_name": "인사이트 시각화"},
        {"agent_name": "Report Generator", "skill_name": "종합 분석 보고서 생성"}
    ]
    
    plan_viz.display_plan(sample_plan, "📊 포괄적 데이터 분석 계획")

def demo_ml_pipeline_plan():
    """ML 파이프라인 계획 시각화"""
    plan_viz = PlanVisualization()
    
    ml_plan = [
        {"agent_name": "Data Preprocessor", "skill_name": "데이터 전처리 및 피처 엔지니어링"},
        {"agent_name": "Model Selector", "skill_name": "최적 모델 선택 및 하이퍼파라미터 튜닝"},
        {"agent_name": "Model Trainer", "skill_name": "모델 학습 및 검증"},
        {"agent_name": "Performance Evaluator", "skill_name": "모델 성능 평가 및 해석"},
        {"agent_name": "Deployment Specialist", "skill_name": "모델 배포 준비"}
    ]
    
    plan_viz.display_plan(ml_plan, "🤖 머신러닝 파이프라인 계획")

def demo_message_translation():
    """메시지 번역 데모"""
    st.header("🔄 사용자 친화적 메시지 번역")
    
    st.markdown("""
    ### 개선 전 vs 개선 후
    
    **개선 전:** 기술적 A2A 프로토콜 메시지가 그대로 노출
    **개선 후:** 자연스러운 언어로 번역된 사용자 친화적 메시지
    """)
    
    # 샘플 메시지들
    sample_messages = {
        "오류 메시지": {
            "messageId": "d5382743-49e1-4938-8f92-28921f14ca2f",
            "parts": [
                {
                    "root": {
                        "text": "❌ **Dataset Not Found: 'titanic.csv'**\n\n**Available datasets:**\n• `sample_sales_data.csv`\n• `customer_data.csv`\n\n**Solution:** Use one of the available dataset IDs above, or upload new data via the Data Loader page."
                    }
                }
            ],
            "response_type": "direct_message"
        },
        "성공 메시지": {
            "messageId": "72620c50-ebeb-4269-9a45-dbfa74b5b5c0",
            "parts": [
                {
                    "root": {
                        "text": "# 📊 Data Analysis Results for `sales_data.csv`\n\nThe analysis has been completed successfully. Here are the key findings:\n\n## Dataset Overview\n- Shape: 1000 rows × 8 columns\n- No missing values detected\n- Data types: 5 numerical, 3 categorical\n\n## Key Insights\n1. Strong correlation between price and sales volume\n2. Seasonal patterns identified in Q4\n3. Regional variations in customer preferences"
                    }
                }
            ],
            "response_type": "direct_message"
        },
        "처리 중 메시지": {
            "messageId": "abc123-def456-ghi789",
            "parts": [
                {
                    "root": {
                        "text": "Processing data analysis request... Current status: Feature engineering in progress. ETA: 2 minutes."
                    }
                }
            ],
            "response_type": "direct_message"
        }
    }
    
    # 메시지 타입 선택
    selected_type = st.selectbox(
        "메시지 타입 선택:",
        list(sample_messages.keys()),
        key="message_type_select"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 개선 전 (기술적)")
        with st.expander("원시 A2A 메시지", expanded=True):
            st.json(sample_messages[selected_type])
    
    with col2:
        st.subheader("✨ 개선 후 (사용자 친화적)")
        renderer = MessageRenderer()
        renderer.render_a2a_message(sample_messages[selected_type])

def demo_beautiful_results():
    """아름다운 결과 표시 데모"""
    st.header("✨ 아름다운 분석 결과 표시")
    
    st.markdown("""
    ### 개선 전 vs 개선 후
    
    **개선 전:** 단순한 텍스트와 기본 마크다운으로 결과 표시
    **개선 후:** 시각적으로 매력적이고 구조화된 결과 표시
    """)
    
    # 샘플 분석 결과들
    sample_results = {
        "데이터 분석 결과": {
            "output_type": "markdown",
            "output": """# 📊 Sales Data Analysis Results

## Dataset Overview
- **Total Records**: 1,000 sales transactions
- **Date Range**: January 2023 - December 2023
- **Columns**: 8 features including price, quantity, region, product

## Key Findings

### 1. Sales Performance
- **Total Revenue**: $2,450,000
- **Average Order Value**: $245
- **Top Performing Month**: December 2023

### 2. Product Analysis
- **Best Seller**: Product A (35% of total sales)
- **Highest Margin**: Product C (45% profit margin)
- **Fastest Growing**: Product D (+120% YoY)

### 3. Regional Insights
- **Top Region**: West Coast (40% of revenue)
- **Fastest Growing**: Southeast (+85% growth)
- **Opportunity**: Northeast (underperforming)

## Recommendations
1. **Inventory Management**: Increase Product A stock for Q1 2024
2. **Marketing Focus**: Expand Product D promotion campaigns
3. **Regional Strategy**: Investigate Northeast market barriers
4. **Seasonal Planning**: Prepare for December peak season

## Next Steps
- Implement dynamic pricing for Product C
- Launch targeted campaigns in Northeast
- Develop Product D expansion strategy
""",
            "agent_name": "Sales Data Analyst"
        },
        "시각화 코드": {
            "output_type": "code",
            "output": """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 매출 트렌드 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales ($)')

# 제품별 매출 분포
plt.subplot(1, 2, 2)
product_sales.plot(kind='bar', color='skyblue')
plt.title('Sales by Product')
plt.xlabel('Product')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()""",
            "agent_name": "Visualization Expert"
        }
    }
    
    # 결과 타입 선택
    selected_result = st.selectbox(
        "결과 타입 선택:",
        list(sample_results.keys()),
        key="result_type_select"
    )
    
    # 결과 표시
    beautiful_results = BeautifulResults()
    result_data = sample_results[selected_result]
    beautiful_results.display_analysis_result(
        result_data, 
        result_data["agent_name"]
    )

# 추가 데모 섹션
def demo_comparison_section():
    """Before/After 비교 섹션"""
    st.header("📊 개선 전후 비교")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("❌ 개선 전")
        st.code("""
{
  "messageId": "d5382743-49e1-4938-8f92-28921f14ca2f",
  "parts": [
    {
      "root": {
        "text": "ValidationError: Dataset 'titanic.csv' not found"
      }
    }
  ],
  "response_type": "direct_message"
}
        """, language="json")
        
        st.markdown("**문제점:**")
        st.markdown("- 기술적 용어 노출")
        st.markdown("- JSON 구조 표시")
        st.markdown("- 사용자 친화적이지 않음")
    
    with col2:
        st.subheader("✅ 개선 후")
        
        # 개선된 메시지 카드 스타일
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #e74c3c15 0%, #e74c3c05 100%);
            border-left: 4px solid #e74c3c;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="font-size: 24px; margin-right: 12px;">📊</div>
                <h4 style="margin: 0; color: #2c3e50;">데이터 분석가</h4>
            </div>
            <div style="color: #2c3e50; line-height: 1.6;">
                <strong>🔍 데이터 분석가의 알림</strong><br><br>
                요청하신 데이터셋을 찾을 수 없었습니다.<br><br>
                <strong>📋 현재 사용 가능한 데이터셋:</strong><br>
                • sample_sales_data.csv<br><br>
                <strong>💡 해결 방법:</strong><br>
                - 위 데이터셋 중 하나를 선택해 주세요<br>
                - 또는 Data Loader 페이지에서 새 데이터를 업로드하세요
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**개선점:**")
        st.markdown("- 자연스러운 언어 사용")
        st.markdown("- 시각적으로 매력적인 디자인")
        st.markdown("- 구체적인 해결책 제시")

if __name__ == "__main__":
    main() 