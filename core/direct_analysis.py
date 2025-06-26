#!/usr/bin/env python3
"""
직접 데이터 분석 모듈
A2A 프로토콜 우회하여 즉시 분석 결과 제공
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from core.data_manager import DataManager

logger = logging.getLogger(__name__)

class DirectAnalysisEngine:
    """A2A 우회 직접 분석 엔진"""
    
    def __init__(self):
        self.data_manager = DataManager()
    
    def perform_comprehensive_analysis(self, prompt: str = "Analyze this dataset", data_id: str = None) -> Dict[str, Any]:
        """포괄적인 데이터 분석 수행"""
        
        try:
            # 사용 가능한 데이터프레임 확인
            available_dfs = self.data_manager.list_dataframes()
            logger.info(f"💾 사용 가능한 데이터프레임: {available_dfs}")
            
            if not available_dfs:
                return {
                    "success": False,
                    "content": """❌ **데이터 없음**

**문제**: 아직 업로드된 데이터셋이 없습니다.

**해결방법:**
1. 🔄 **데이터 로더** 페이지로 이동
2. 📁 CSV, Excel 등의 데이터 파일 업로드  
3. 📊 다시 돌아와서 데이터 분석 요청

**현재 사용 가능한 데이터셋**: 없음
""",
                    "response_type": "error"
                }
            
            # 첫 번째 데이터프레임 사용
            df_id = data_id if data_id and data_id in available_dfs else available_dfs[0]
            df = self.data_manager.get_dataframe(df_id)
            
            if df is None:
                return {
                    "success": False,
                    "content": f"❌ 데이터셋 '{df_id}'를 로드할 수 없습니다.",
                    "response_type": "error"
                }
            
            # 실제 분석 수행
            analysis_result = self._generate_analysis_report(df, df_id, prompt)
            
            return {
                "success": True,
                "content": analysis_result,
                "response_type": "analysis",
                "data_id": df_id,
                "data_shape": df.shape
            }
            
        except Exception as e:
            logger.error(f"❌ 분석 실패: {e}", exc_info=True)
            return {
                "success": False,
                "content": f"❌ 분석 실패: {str(e)}",
                "response_type": "error"
            }
    
    def _generate_analysis_report(self, df: pd.DataFrame, df_id: str, prompt: str) -> str:
        """상세한 분석 보고서 생성"""
        
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
            stats_df = df.describe().round(2)
            stats_table = "| 통계 | " + " | ".join(numeric_cols[:5]) + " |\n"
            stats_table += "|------|" + "------|" * min(len(numeric_cols), 5) + "\n"
            
            for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                if stat in stats_df.index:
                    row = f"| **{stat}** |"
                    for col in numeric_cols[:5]:
                        if col in stats_df.columns:
                            value = stats_df.loc[stat, col]
                            row += f" {value:.2f} |"
                    stats_table += row + "\n"
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
            if col in df.columns:
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
        
        # 도메인별 특화 분석
        domain_analysis = self._generate_domain_analysis(df)
        
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

{domain_analysis}

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
🔧 **분석 엔진**: Direct Analysis Engine (우회 모드)
"""
        
        return final_result
    
    def _generate_domain_analysis(self, df: pd.DataFrame) -> str:
        """도메인별 특화 분석"""
        
        # Titanic 데이터 특화 분석
        if 'Survived' in df.columns:
            survival_rate = df['Survived'].mean() * 100
            analysis = f"""
### ⚓ 생존율 분석

**전체 생존율**: {survival_rate:.1f}%

**성별별 생존율**:
"""
            if 'Sex' in df.columns:
                sex_survival = df.groupby('Sex')['Survived'].mean() * 100
                for sex, rate in sex_survival.items():
                    analysis += f"- {sex}: {rate:.1f}%\n"
            
            if 'Pclass' in df.columns:
                analysis += "\n**객실 등급별 생존율**:\n"
                class_survival = df.groupby('Pclass')['Survived'].mean() * 100
                for pclass, rate in class_survival.items():
                    analysis += f"- {pclass}등석: {rate:.1f}%\n"
            
            return analysis
        
        # 일반적인 도메인 분석
        return "### 📈 추가 분석\n일반적인 데이터셋으로 추가적인 도메인별 분석을 제공할 수 있습니다."
    
    def analyze_with_fallback(self, prompt: str, data_id: str = None) -> Dict[str, Any]:
        """A2A 실패 시 사용할 대체 분석 메서드"""
        logger.info("🔄 Direct Analysis Engine으로 분석을 진행합니다")
        return self.perform_comprehensive_analysis(prompt, data_id)

# 글로벌 인스턴스
direct_analysis_engine = DirectAnalysisEngine() 