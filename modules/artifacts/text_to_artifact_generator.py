"""
텍스트 응답에서 아티팩트 생성기

A2A 에이전트가 텍스트만 반환하는 문제를 해결하기 위해
텍스트 응답을 분석하여 적절한 아티팩트를 생성하는 모듈
"""

import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum

# ArtifactType definition
class ArtifactType(Enum):
    """아티팩트 유형"""
    PLOTLY_CHART = "plotly_chart"
    DATAFRAME = "dataframe"
    IMAGE = "image"
    CODE = "code"
    TEXT = "text"

@dataclass
class ArtifactInfo:
    """아티팩트 정보 클래스"""
    artifact_id: str
    type: ArtifactType
    title: str
    data: Any
    agent_id: str
    created_at: datetime
    metadata: Dict[str, Any]

logger = logging.getLogger(__name__)

class TextToArtifactGenerator:
    """텍스트 응답에서 아티팩트를 생성하는 클래스"""
    
    def __init__(self):
        self.patterns = {
            # 상관관계 분석 패턴
            'correlation': [
                r'상관관계|correlation|피어슨|spearman|kendall',
                r'관련성|연관성|관계',
                r'양의 관계|음의 관계|positive|negative correlation'
            ],
            
            # 기술 통계 패턴
            'statistics': [
                r'평균|mean|average',
                r'표준편차|standard deviation|std',
                r'중앙값|median',
                r'최댓값|최솟값|max|min',
                r'분산|variance',
                r'기술통계|descriptive statistics'
            ],
            
            # 분포 분석 패턴
            'distribution': [
                r'분포|distribution',
                r'히스토그램|histogram',
                r'정규분포|normal distribution',
                r'왜도|skewness',
                r'첨도|kurtosis'
            ],
            
            # 시계열 분석 패턴
            'timeseries': [
                r'시계열|time series',
                r'트렌드|trend',
                r'계절성|seasonality',
                r'시간|time|date|날짜'
            ],
            
            # 이상치 분석 패턴
            'outliers': [
                r'이상치|outlier',
                r'특이값|anomaly',
                r'극값|extreme'
            ],
            
            # 범주형 분석 패턴
            'categorical': [
                r'범주|category|categorical',
                r'빈도|frequency',
                r'카이제곱|chi-square',
                r'그룹별|group by'
            ]
        }
    
    def generate_artifacts_from_text(
        self, 
        text_response: str, 
        dataset: Optional[pd.DataFrame] = None,
        agent_id: str = "text_agent",
        analysis_type: str = "general"
    ) -> List[ArtifactInfo]:
        """
        텍스트 응답에서 아티팩트를 생성
        
        Args:
            text_response: 에이전트의 텍스트 응답
            dataset: 분석에 사용된 데이터셋 (선택적)
            agent_id: 에이전트 ID
            analysis_type: 분석 유형
        
        Returns:
            생성된 아티팩트 목록
        """
        
        artifacts = []
        detected_patterns = self._detect_analysis_patterns(text_response)
        
        logger.info(f"감지된 분석 패턴: {detected_patterns}")
        
        # 데이터셋이 제공된 경우에만 실제 차트 생성
        if dataset is not None and not dataset.empty:
            for pattern in detected_patterns:
                if pattern == 'correlation':
                    artifact = self._generate_correlation_chart(dataset, agent_id, text_response)
                    if artifact:
                        artifacts.append(artifact)
                
                elif pattern == 'statistics':
                    artifact = self._generate_statistics_table(dataset, agent_id, text_response)
                    if artifact:
                        artifacts.append(artifact)
                
                elif pattern == 'distribution':
                    artifact = self._generate_distribution_chart(dataset, agent_id, text_response)
                    if artifact:
                        artifacts.append(artifact)
                
                elif pattern == 'timeseries':
                    artifact = self._generate_timeseries_chart(dataset, agent_id, text_response)
                    if artifact:
                        artifacts.append(artifact)
                
                elif pattern == 'categorical':
                    artifact = self._generate_categorical_chart(dataset, agent_id, text_response)
                    if artifact:
                        artifacts.append(artifact)
        
        # 텍스트 응답 자체도 아티팩트로 추가
        text_artifact = self._generate_text_artifact(text_response, agent_id, analysis_type)
        artifacts.append(text_artifact)
        
        return artifacts
    
    def _detect_analysis_patterns(self, text: str) -> List[str]:
        """텍스트에서 분석 패턴을 감지"""
        detected = []
        text_lower = text.lower()
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if pattern_type not in detected:
                        detected.append(pattern_type)
                    break
        
        return detected
    
    def _generate_correlation_chart(
        self, 
        dataset: pd.DataFrame, 
        agent_id: str, 
        text_response: str
    ) -> Optional[ArtifactInfo]:
        """상관관계 히트맵 차트 생성"""
        try:
            # 숫자형 컬럼만 선택
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return None
            
            # 상관관계 계산
            correlation_matrix = dataset[numeric_cols].corr()
            
            # Plotly 히트맵 생성
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="변수 간 상관관계 분석",
                xaxis_title="변수",
                yaxis_title="변수",
                width=600,
                height=500
            )
            
            return ArtifactInfo(
                artifact_id=f"correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=ArtifactType.PLOTLY_CHART,
                title="변수 간 상관관계 히트맵",
                data=fig.to_dict(),
                agent_id=agent_id,
                created_at=datetime.now(),
                metadata={
                    "chart_type": "heatmap",
                    "analysis_type": "correlation",
                    "variables": list(numeric_cols),
                    "generated_from": "text_response"
                }
            )
            
        except Exception as e:
            logger.error(f"상관관계 차트 생성 오류: {e}")
            return None
    
    def _generate_statistics_table(
        self, 
        dataset: pd.DataFrame, 
        agent_id: str, 
        text_response: str
    ) -> Optional[ArtifactInfo]:
        """기술 통계 테이블 생성"""
        try:
            # 숫자형 컬럼만 선택
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            # 기술 통계 계산
            stats = dataset[numeric_cols].describe()
            stats = stats.round(3)
            
            # DataFrame을 아티팩트 형식으로 변환
            table_data = {
                "columns": ["통계량"] + list(stats.columns),
                "data": []
            }
            
            # 행별로 데이터 추가
            for idx in stats.index:
                row = [idx] + list(stats.loc[idx].values)
                table_data["data"].append(row)
            
            table_data["index"] = list(range(len(table_data["data"])))
            
            return ArtifactInfo(
                artifact_id=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=ArtifactType.DATAFRAME,
                title="기술 통계 요약",
                data=table_data,
                agent_id=agent_id,
                created_at=datetime.now(),
                metadata={
                    "table_type": "descriptive_statistics",
                    "analysis_type": "statistics",
                    "variables": list(numeric_cols),
                    "generated_from": "text_response"
                }
            )
            
        except Exception as e:
            logger.error(f"통계 테이블 생성 오류: {e}")
            return None
    
    def _generate_distribution_chart(
        self, 
        dataset: pd.DataFrame, 
        agent_id: str, 
        text_response: str
    ) -> Optional[ArtifactInfo]:
        """분포 차트 생성"""
        try:
            # 숫자형 컬럼 선택
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            # 첫 번째 숫자형 컬럼 사용
            col = numeric_cols[0]
            
            # 히스토그램 생성
            fig = px.histogram(
                dataset, 
                x=col,
                title=f"{col} 분포 분석",
                nbins=30,
                marginal="box"  # 박스플롯 추가
            )
            
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="빈도",
                showlegend=False,
                width=600,
                height=400
            )
            
            return ArtifactInfo(
                artifact_id=f"distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=ArtifactType.PLOTLY_CHART,
                title=f"{col} 분포 히스토그램",
                data=fig.to_dict(),
                agent_id=agent_id,
                created_at=datetime.now(),
                metadata={
                    "chart_type": "histogram",
                    "analysis_type": "distribution",
                    "variable": col,
                    "generated_from": "text_response"
                }
            )
            
        except Exception as e:
            logger.error(f"분포 차트 생성 오류: {e}")
            return None
    
    def _generate_timeseries_chart(
        self, 
        dataset: pd.DataFrame, 
        agent_id: str, 
        text_response: str
    ) -> Optional[ArtifactInfo]:
        """시계열 차트 생성"""
        try:
            # 날짜/시간 컬럼 찾기
            date_cols = dataset.select_dtypes(include=['datetime64', 'object']).columns
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            
            if len(date_cols) == 0 or len(numeric_cols) == 0:
                return None
            
            # 시간 축으로 사용할 컬럼 (인덱스 또는 첫 번째 컬럼)
            if hasattr(dataset.index, 'dtype') and 'datetime' in str(dataset.index.dtype):
                x_data = dataset.index
                y_data = dataset[numeric_cols[0]]
                x_title = "Date"
            else:
                x_data = range(len(dataset))
                y_data = dataset[numeric_cols[0]]
                x_title = "Index"
            
            # 선 그래프 생성
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',
                name=numeric_cols[0],
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f"{numeric_cols[0]} 시계열 분석",
                xaxis_title=x_title,
                yaxis_title=numeric_cols[0],
                width=700,
                height=400
            )
            
            return ArtifactInfo(
                artifact_id=f"timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=ArtifactType.PLOTLY_CHART,
                title=f"{numeric_cols[0]} 시계열 차트",
                data=fig.to_dict(),
                agent_id=agent_id,
                created_at=datetime.now(),
                metadata={
                    "chart_type": "line",
                    "analysis_type": "timeseries",
                    "variable": numeric_cols[0],
                    "generated_from": "text_response"
                }
            )
            
        except Exception as e:
            logger.error(f"시계열 차트 생성 오류: {e}")
            return None
    
    def _generate_categorical_chart(
        self, 
        dataset: pd.DataFrame, 
        agent_id: str, 
        text_response: str
    ) -> Optional[ArtifactInfo]:
        """범주형 데이터 차트 생성"""
        try:
            # 범주형/문자열 컬럼 찾기
            cat_cols = dataset.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) == 0:
                return None
            
            # 첫 번째 범주형 컬럼 사용
            col = cat_cols[0]
            
            # 값 개수 계산
            value_counts = dataset[col].value_counts()
            
            # 막대 차트 생성
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"{col} 범주별 빈도 분석",
                labels={'x': col, 'y': '빈도'}
            )
            
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="빈도",
                width=600,
                height=400
            )
            
            return ArtifactInfo(
                artifact_id=f"categorical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=ArtifactType.PLOTLY_CHART,
                title=f"{col} 범주별 분포",
                data=fig.to_dict(),
                agent_id=agent_id,
                created_at=datetime.now(),
                metadata={
                    "chart_type": "bar",
                    "analysis_type": "categorical",
                    "variable": col,
                    "generated_from": "text_response"
                }
            )
            
        except Exception as e:
            logger.error(f"범주형 차트 생성 오류: {e}")
            return None
    
    def _generate_text_artifact(
        self, 
        text_response: str, 
        agent_id: str, 
        analysis_type: str
    ) -> ArtifactInfo:
        """텍스트 응답을 아티팩트로 변환"""
        
        # 텍스트를 마크다운 형식으로 정리
        cleaned_text = self._clean_text_response(text_response)
        
        return ArtifactInfo(
            artifact_id=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=ArtifactType.TEXT,
            title=f"{analysis_type.title()} 분석 보고서",
            data=cleaned_text,
            agent_id=agent_id,
            created_at=datetime.now(),
            metadata={
                "format": "markdown",
                "analysis_type": analysis_type,
                "word_count": len(cleaned_text.split()),
                "generated_from": "text_response"
            }
        )
    
    def _clean_text_response(self, text: str) -> str:
        """텍스트 응답을 정리하여 마크다운 형식으로 변환"""
        
        # 이모지 헤더를 마크다운 헤더로 변환
        text = re.sub(r'## (🎯|📊|💡|📈|📝|🔍|✅|⚡)', r'## ', text)
        text = re.sub(r'### (🎯|📊|💡|📈|📝|🔍|✅|⚡)', r'### ', text)
        
        # 불필요한 HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 시작과 끝 공백 제거
        text = text.strip()
        
        return text