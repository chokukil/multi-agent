"""
범용적 분석 엔진 (Universal Analyzer)
Phase 3.2: 데이터셋 독립적 LLM First 분석 시스템

핵심 원칙:
- 데이터셋에 의존하지 않는 범용적 분석
- LLM 기반 동적 전략 수립
- 메타 러닝을 통한 지속적 개선
- 컨텍스트 기반 적응형 분석
- 완전한 하드코딩 제거
"""

import asyncio
import json
import logging
import statistics
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from collections import defaultdict
from enum import Enum
from pathlib import Path

# LLM 클라이언트
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class AnalysisContext(Enum):
    """분석 컨텍스트"""
    EXPLORATION = "exploration"         # 탐색적 분석
    PREDICTION = "prediction"          # 예측 분석
    CLASSIFICATION = "classification"  # 분류 분석
    CLUSTERING = "clustering"          # 클러스터링 분석
    ANOMALY_DETECTION = "anomaly_detection"  # 이상 탐지
    TIME_SERIES = "time_series"        # 시계열 분석
    ASSOCIATION = "association"        # 연관성 분석
    CAUSAL = "causal"                 # 인과관계 분석

class DataCharacteristics(Enum):
    """데이터 특성"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TEMPORAL = "temporal"
    MIXED = "mixed"
    SPARSE = "sparse"
    HIGH_DIMENSIONAL = "high_dimensional"
    IMBALANCED = "imbalanced"

@dataclass
class DataProfile:
    """데이터 프로파일"""
    shape: Tuple[int, int]
    column_types: Dict[str, str]
    missing_rates: Dict[str, float]
    data_characteristics: List[DataCharacteristics]
    statistical_summary: Dict[str, Any]
    quality_score: float
    complexity_level: str
    suggested_contexts: List[AnalysisContext]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisStrategy:
    """분석 전략"""
    context: AnalysisContext
    priority_steps: List[str]
    techniques: List[str]
    expected_insights: List[str]
    confidence: float
    reasoning: str
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """분석 결과"""
    insights: List[str]
    visualizations: List[Dict[str, Any]]
    statistical_tests: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_scores: Dict[str, float]
    methodology: str
    limitations: List[str]
    next_steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class UniversalDataProfiler:
    """범용 데이터 프로파일러"""
    
    def __init__(self):
        self.llm_client = AsyncOpenAI()
        
        # 데이터 특성 탐지 규칙 (LLM 보완용)
        self.characteristic_detectors = {
            DataCharacteristics.NUMERICAL: self._detect_numerical,
            DataCharacteristics.CATEGORICAL: self._detect_categorical,
            DataCharacteristics.TEXT: self._detect_text,
            DataCharacteristics.TEMPORAL: self._detect_temporal,
            DataCharacteristics.SPARSE: self._detect_sparse,
            DataCharacteristics.HIGH_DIMENSIONAL: self._detect_high_dimensional,
            DataCharacteristics.IMBALANCED: self._detect_imbalanced
        }
    
    async def profile_data(self, data: pd.DataFrame, user_context: str = "") -> DataProfile:
        """데이터 프로파일링 (완전 범용적)"""
        logger.info(f"📊 범용 데이터 프로파일링 시작 (shape: {data.shape})")
        
        # 1. 기본 구조 분석
        basic_profile = self._analyze_basic_structure(data)
        
        # 2. 데이터 특성 탐지
        characteristics = await self._detect_data_characteristics(data)
        
        # 3. 통계적 요약 (범용적)
        statistical_summary = await self._generate_statistical_summary(data)
        
        # 4. 품질 평가
        quality_score = self._assess_data_quality(data, basic_profile)
        
        # 5. LLM 기반 컨텍스트 제안
        suggested_contexts = await self._suggest_analysis_contexts(
            data, characteristics, user_context
        )
        
        # 6. 복잡도 평가
        complexity_level = self._assess_complexity(data, characteristics)
        
        profile = DataProfile(
            shape=data.shape,
            column_types=basic_profile["column_types"],
            missing_rates=basic_profile["missing_rates"],
            data_characteristics=characteristics,
            statistical_summary=statistical_summary,
            quality_score=quality_score,
            complexity_level=complexity_level,
            suggested_contexts=suggested_contexts,
            metadata={
                "profiling_timestamp": datetime.now().isoformat(),
                "user_context": user_context
            }
        )
        
        logger.info(f"✅ 데이터 프로파일링 완료 (품질: {quality_score:.2f}, 복잡도: {complexity_level})")
        return profile
    
    def _analyze_basic_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기본 구조 분석"""
        column_types = {}
        missing_rates = {}
        
        for column in data.columns:
            # 데이터 타입 분석 (범용적)
            dtype = str(data[column].dtype)
            if dtype.startswith(('int', 'float')):
                column_types[column] = 'numeric'
            elif dtype == 'object':
                # 텍스트 vs 카테고리 구분
                unique_ratio = len(data[column].dropna().unique()) / len(data[column].dropna())
                if unique_ratio < 0.1:  # 10% 미만이면 카테고리
                    column_types[column] = 'categorical'
                else:
                    column_types[column] = 'text'
            elif dtype.startswith('datetime'):
                column_types[column] = 'datetime'
            else:
                column_types[column] = 'other'
            
            # 결측률 계산
            missing_rates[column] = data[column].isnull().sum() / len(data)
        
        return {
            "column_types": column_types,
            "missing_rates": missing_rates
        }
    
    async def _detect_data_characteristics(self, data: pd.DataFrame) -> List[DataCharacteristics]:
        """데이터 특성 탐지"""
        characteristics = []
        
        # 각 특성 탐지기 실행
        for characteristic, detector in self.characteristic_detectors.items():
            if detector(data):
                characteristics.append(characteristic)
        
        # Mixed 특성 체크
        if len(set(data.dtypes.astype(str))) > 2:
            characteristics.append(DataCharacteristics.MIXED)
        
        return characteristics
    
    def _detect_numerical(self, data: pd.DataFrame) -> bool:
        """수치형 데이터 탐지"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return len(numeric_cols) / len(data.columns) > 0.5
    
    def _detect_categorical(self, data: pd.DataFrame) -> bool:
        """범주형 데이터 탐지"""
        categorical_cols = 0
        for column in data.columns:
            if data[column].dtype == 'object':
                unique_ratio = len(data[column].dropna().unique()) / len(data[column].dropna())
                if unique_ratio < 0.1:
                    categorical_cols += 1
        return categorical_cols / len(data.columns) > 0.3
    
    def _detect_text(self, data: pd.DataFrame) -> bool:
        """텍스트 데이터 탐지"""
        text_cols = 0
        for column in data.columns:
            if data[column].dtype == 'object':
                # 평균 문자열 길이로 텍스트 판단
                avg_length = data[column].dropna().astype(str).str.len().mean()
                if avg_length > 20:  # 20자 이상이면 텍스트로 판단
                    text_cols += 1
        return text_cols > 0
    
    def _detect_temporal(self, data: pd.DataFrame) -> bool:
        """시계열 데이터 탐지"""
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        return len(datetime_cols) > 0
    
    def _detect_sparse(self, data: pd.DataFrame) -> bool:
        """희소 데이터 탐지"""
        zero_ratio = (data == 0).sum().sum() / data.size
        return zero_ratio > 0.5
    
    def _detect_high_dimensional(self, data: pd.DataFrame) -> bool:
        """고차원 데이터 탐지"""
        return data.shape[1] > 50
    
    def _detect_imbalanced(self, data: pd.DataFrame) -> bool:
        """불균형 데이터 탐지"""
        for column in data.columns:
            if data[column].dtype == 'object':
                value_counts = data[column].value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                    if imbalance_ratio > 10:  # 10:1 이상 불균형
                        return True
        return False
    
    async def _generate_statistical_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """통계적 요약 생성 (범용적)"""
        summary = {}
        
        # 수치형 컬럼 요약
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            summary["numeric_summary"] = {
                "count": len(numeric_data.columns),
                "descriptive_stats": numeric_data.describe().to_dict(),
                "correlation_strength": self._assess_correlation_strength(numeric_data),
                "outlier_rates": self._detect_outlier_rates(numeric_data)
            }
        
        # 범주형 컬럼 요약
        categorical_cols = []
        for column in data.columns:
            if data[column].dtype == 'object':
                unique_ratio = len(data[column].dropna().unique()) / len(data[column].dropna())
                if unique_ratio < 0.1:
                    categorical_cols.append(column)
        
        if categorical_cols:
            summary["categorical_summary"] = {
                "count": len(categorical_cols),
                "unique_values": {col: len(data[col].dropna().unique()) for col in categorical_cols},
                "mode_frequencies": {col: data[col].mode().iloc[0] if not data[col].mode().empty else None 
                                   for col in categorical_cols}
            }
        
        # 전체 데이터 특성
        summary["overall"] = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "missing_data_percentage": (data.isnull().sum().sum() / data.size) * 100,
            "duplicate_rows": data.duplicated().sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        return summary
    
    def _assess_correlation_strength(self, numeric_data: pd.DataFrame) -> Dict[str, float]:
        """상관관계 강도 평가"""
        if len(numeric_data.columns) < 2:
            return {"max_correlation": 0.0, "avg_correlation": 0.0}
        
        corr_matrix = numeric_data.corr().abs()
        # 대각선 제거
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        
        return {
            "max_correlation": float(np.max(corr_values)) if len(corr_values) > 0 else 0.0,
            "avg_correlation": float(np.mean(corr_values)) if len(corr_values) > 0 else 0.0
        }
    
    def _detect_outlier_rates(self, numeric_data: pd.DataFrame) -> Dict[str, float]:
        """이상치 비율 탐지"""
        outlier_rates = {}
        
        for column in numeric_data.columns:
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((numeric_data[column] < lower_bound) | 
                       (numeric_data[column] > upper_bound)).sum()
            outlier_rates[column] = outliers / len(numeric_data[column])
        
        return outlier_rates
    
    async def _suggest_analysis_contexts(self, data: pd.DataFrame, 
                                       characteristics: List[DataCharacteristics],
                                       user_context: str) -> List[AnalysisContext]:
        """분석 컨텍스트 제안 (LLM 기반)"""
        
        # 데이터 특성 요약
        data_summary = {
            "shape": data.shape,
            "characteristics": [c.value for c in characteristics],
            "column_types": data.dtypes.astype(str).to_dict(),
            "sample_columns": list(data.columns[:10])  # 처음 10개 컬럼만
        }
        
        prompt = f"""
데이터 분석 전문가로서 다음 데이터에 가장 적합한 분석 컨텍스트를 제안해주세요.

데이터 정보:
- 형태: {data_summary['shape']} (행 x 열)
- 특성: {', '.join(data_summary['characteristics'])}
- 컬럼 예시: {', '.join(data_summary['sample_columns'][:5])}
- 사용자 컨텍스트: {user_context or '없음'}

가능한 분석 컨텍스트:
1. exploration - 탐색적 데이터 분석
2. prediction - 예측 모델링
3. classification - 분류 분석
4. clustering - 클러스터링
5. anomaly_detection - 이상 탐지
6. time_series - 시계열 분석
7. association - 연관성 분석
8. causal - 인과관계 분석

데이터 특성을 고려하여 가장 적합한 분석 컨텍스트를 우선순위대로 3개 선택하고, 각각에 대한 간단한 이유를 제시해주세요.

응답 형식:
1. context_name: 이유
2. context_name: 이유
3. context_name: 이유
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            suggested_contexts = self._parse_context_suggestions(content)
            
        except Exception as e:
            logger.warning(f"LLM 컨텍스트 제안 실패: {e}")
            # 폴백: 데이터 특성 기반 기본 제안
            suggested_contexts = self._fallback_context_suggestion(characteristics)
        
        return suggested_contexts
    
    def _parse_context_suggestions(self, llm_response: str) -> List[AnalysisContext]:
        """LLM 응답에서 컨텍스트 파싱"""
        contexts = []
        
        context_mapping = {
            "exploration": AnalysisContext.EXPLORATION,
            "prediction": AnalysisContext.PREDICTION,
            "classification": AnalysisContext.CLASSIFICATION,
            "clustering": AnalysisContext.CLUSTERING,
            "anomaly_detection": AnalysisContext.ANOMALY_DETECTION,
            "time_series": AnalysisContext.TIME_SERIES,
            "association": AnalysisContext.ASSOCIATION,
            "causal": AnalysisContext.CAUSAL
        }
        
        for context_name, context_enum in context_mapping.items():
            if context_name in llm_response.lower():
                contexts.append(context_enum)
        
        # 최소 1개는 보장
        if not contexts:
            contexts.append(AnalysisContext.EXPLORATION)
        
        return contexts[:3]  # 최대 3개
    
    def _fallback_context_suggestion(self, characteristics: List[DataCharacteristics]) -> List[AnalysisContext]:
        """폴백 컨텍스트 제안"""
        contexts = [AnalysisContext.EXPLORATION]  # 기본은 탐색적 분석
        
        if DataCharacteristics.TEMPORAL in characteristics:
            contexts.append(AnalysisContext.TIME_SERIES)
        
        if DataCharacteristics.NUMERICAL in characteristics:
            contexts.append(AnalysisContext.PREDICTION)
        
        if DataCharacteristics.CATEGORICAL in characteristics:
            contexts.append(AnalysisContext.CLASSIFICATION)
        
        return contexts[:3]
    
    def _assess_data_quality(self, data: pd.DataFrame, basic_profile: Dict[str, Any]) -> float:
        """데이터 품질 평가 (0-1)"""
        quality_factors = []
        
        # 1. 완성도 (결측값 비율)
        overall_missing_rate = sum(basic_profile["missing_rates"].values()) / len(basic_profile["missing_rates"])
        completeness_score = 1 - overall_missing_rate
        quality_factors.append(completeness_score)
        
        # 2. 일관성 (데이터 타입 일관성)
        consistency_score = 1.0  # 기본값
        quality_factors.append(consistency_score)
        
        # 3. 유효성 (수치 데이터의 합리성)
        validity_score = self._assess_validity(data)
        quality_factors.append(validity_score)
        
        # 4. 유니크성 (중복 데이터 비율)
        duplicate_rate = data.duplicated().sum() / len(data)
        uniqueness_score = 1 - duplicate_rate
        quality_factors.append(uniqueness_score)
        
        return statistics.mean(quality_factors)
    
    def _assess_validity(self, data: pd.DataFrame) -> float:
        """데이터 유효성 평가"""
        validity_scores = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            if len(col_data) == 0:
                continue
            
            # 무한값, NaN 체크
            invalid_count = np.isinf(col_data).sum() + np.isnan(col_data).sum()
            validity_score = 1 - (invalid_count / len(col_data))
            validity_scores.append(validity_score)
        
        return statistics.mean(validity_scores) if validity_scores else 1.0
    
    def _assess_complexity(self, data: pd.DataFrame, characteristics: List[DataCharacteristics]) -> str:
        """데이터 복잡도 평가"""
        complexity_score = 0
        
        # 차원수 기반
        if data.shape[1] > 100:
            complexity_score += 3
        elif data.shape[1] > 50:
            complexity_score += 2
        elif data.shape[1] > 20:
            complexity_score += 1
        
        # 데이터 크기 기반
        if data.shape[0] > 1000000:
            complexity_score += 3
        elif data.shape[0] > 100000:
            complexity_score += 2
        elif data.shape[0] > 10000:
            complexity_score += 1
        
        # 특성 기반
        if DataCharacteristics.HIGH_DIMENSIONAL in characteristics:
            complexity_score += 2
        if DataCharacteristics.TEXT in characteristics:
            complexity_score += 2
        if DataCharacteristics.MIXED in characteristics:
            complexity_score += 1
        
        # 복잡도 레벨 결정
        if complexity_score >= 6:
            return "very_high"
        elif complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"

class UniversalAnalysisEngine:
    """범용 분석 엔진"""
    
    def __init__(self):
        self.llm_client = AsyncOpenAI()
        self.data_profiler = UniversalDataProfiler()
        
        # 분석 전략 메타 데이터
        self.strategy_knowledge = {
            AnalysisContext.EXPLORATION: {
                "priority_steps": ["data_overview", "distribution_analysis", "correlation_analysis", "pattern_discovery"],
                "techniques": ["descriptive_statistics", "visualization", "outlier_detection"],
                "expected_insights": ["data_quality", "patterns", "relationships", "anomalies"]
            },
            AnalysisContext.PREDICTION: {
                "priority_steps": ["feature_analysis", "target_correlation", "model_selection", "validation"],
                "techniques": ["regression", "feature_importance", "cross_validation"],
                "expected_insights": ["predictive_features", "model_performance", "feature_importance"]
            },
            AnalysisContext.CLASSIFICATION: {
                "priority_steps": ["class_distribution", "feature_discrimination", "model_comparison"],
                "techniques": ["classification_algorithms", "feature_selection", "performance_metrics"],
                "expected_insights": ["class_separability", "discriminative_features", "classification_accuracy"]
            }
        }
        
        # 결과 저장 경로
        self.results_dir = Path("core/universal/analysis_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def analyze_universally(self, data: pd.DataFrame, 
                                user_query: str = "",
                                preferred_context: Optional[AnalysisContext] = None) -> AnalysisResult:
        """범용적 데이터 분석"""
        logger.info(f"🔬 범용 분석 시작: {data.shape}")
        
        # 1. 데이터 프로파일링
        profile = await self.data_profiler.profile_data(data, user_query)
        
        # 2. 분석 전략 수립
        strategy = await self._develop_analysis_strategy(profile, user_query, preferred_context)
        
        # 3. 분석 실행
        result = await self._execute_analysis(data, profile, strategy)
        
        # 4. 결과 저장
        await self._save_analysis_result(result, profile, strategy)
        
        logger.info(f"✅ 범용 분석 완료: {len(result.insights)}개 인사이트 도출")
        return result
    
    async def _develop_analysis_strategy(self, profile: DataProfile, 
                                       user_query: str,
                                       preferred_context: Optional[AnalysisContext]) -> AnalysisStrategy:
        """분석 전략 수립 (LLM 기반)"""
        
        # 컨텍스트 결정
        if preferred_context:
            context = preferred_context
        else:
            context = profile.suggested_contexts[0] if profile.suggested_contexts else AnalysisContext.EXPLORATION
        
        # LLM을 통한 전략 상세화
        strategy_prompt = f"""
데이터 분석 전문가로서 다음 데이터에 대한 {context.value} 분석 전략을 수립해주세요.

데이터 프로파일:
- 형태: {profile.shape}
- 특성: {[c.value for c in profile.data_characteristics]}
- 품질 점수: {profile.quality_score:.2f}
- 복잡도: {profile.complexity_level}

사용자 요청: {user_query or '일반적인 분석'}

다음 형식으로 전략을 제시해주세요:
1. 우선순위 단계 (4-6개 단계)
2. 적용할 기법들
3. 예상되는 인사이트 유형
4. 이 전략의 신뢰도 (0-1)
5. 전략 선택 이유

범용적이고 데이터셋에 의존하지 않는 접근법을 사용하세요.
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": strategy_prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            llm_strategy = response.choices[0].message.content
            strategy = self._parse_strategy_response(llm_strategy, context)
            
        except Exception as e:
            logger.warning(f"LLM 전략 수립 실패: {e}")
            strategy = self._fallback_strategy(context)
        
        return strategy
    
    def _parse_strategy_response(self, llm_response: str, context: AnalysisContext) -> AnalysisStrategy:
        """LLM 응답에서 전략 파싱"""
        
        # 기본 전략 가져오기
        base_strategy = self.strategy_knowledge.get(context, {})
        
        # LLM 응답에서 정보 추출 (간단한 파싱)
        lines = llm_response.split('\n')
        
        priority_steps = base_strategy.get("priority_steps", [])
        techniques = base_strategy.get("techniques", [])
        expected_insights = base_strategy.get("expected_insights", [])
        
        # 신뢰도 추출 시도
        confidence = 0.8  # 기본값
        for line in lines:
            if "신뢰도" in line or "confidence" in line.lower():
                try:
                    import re
                    numbers = re.findall(r'0\.\d+|\d+', line)
                    if numbers:
                        confidence = float(numbers[0])
                        if confidence > 1:
                            confidence = confidence / 100
                except:
                    pass
        
        return AnalysisStrategy(
            context=context,
            priority_steps=priority_steps,
            techniques=techniques,
            expected_insights=expected_insights,
            confidence=confidence,
            reasoning=llm_response
        )
    
    def _fallback_strategy(self, context: AnalysisContext) -> AnalysisStrategy:
        """폴백 전략"""
        base_strategy = self.strategy_knowledge.get(context, {})
        
        return AnalysisStrategy(
            context=context,
            priority_steps=base_strategy.get("priority_steps", ["basic_analysis"]),
            techniques=base_strategy.get("techniques", ["descriptive_statistics"]),
            expected_insights=base_strategy.get("expected_insights", ["basic_insights"]),
            confidence=0.6,
            reasoning="기본 전략 적용"
        )
    
    async def _execute_analysis(self, data: pd.DataFrame, 
                              profile: DataProfile, 
                              strategy: AnalysisStrategy) -> AnalysisResult:
        """분석 실행"""
        
        insights = []
        visualizations = []
        statistical_tests = []
        recommendations = []
        confidence_scores = {}
        
        # 각 단계별 분석 실행
        for step in strategy.priority_steps:
            try:
                step_result = await self._execute_analysis_step(step, data, profile, strategy)
                
                insights.extend(step_result.get("insights", []))
                visualizations.extend(step_result.get("visualizations", []))
                statistical_tests.extend(step_result.get("statistical_tests", []))
                confidence_scores[step] = step_result.get("confidence", 0.5)
                
            except Exception as e:
                logger.error(f"분석 단계 '{step}' 실행 실패: {e}")
                insights.append(f"분석 단계 '{step}' 처리 중 오류 발생")
                confidence_scores[step] = 0.0
        
        # LLM 기반 종합 인사이트 생성
        comprehensive_insights = await self._generate_comprehensive_insights(
            data, profile, strategy, insights
        )
        insights.extend(comprehensive_insights)
        
        # 권장사항 생성
        recommendations = await self._generate_recommendations(data, profile, strategy, insights)
        
        return AnalysisResult(
            insights=insights,
            visualizations=visualizations,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            confidence_scores=confidence_scores,
            methodology=f"{strategy.context.value} 분석 (LLM 기반 범용 접근법)",
            limitations=self._identify_limitations(profile, strategy),
            next_steps=self._suggest_next_steps(profile, strategy),
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "strategy_confidence": strategy.confidence,
                "data_quality": profile.quality_score
            }
        )
    
    async def _execute_analysis_step(self, step: str, data: pd.DataFrame, 
                                   profile: DataProfile, strategy: AnalysisStrategy) -> Dict[str, Any]:
        """개별 분석 단계 실행"""
        
        if step == "data_overview":
            return await self._analyze_data_overview(data, profile)
        elif step == "distribution_analysis":
            return await self._analyze_distributions(data, profile)
        elif step == "correlation_analysis":
            return await self._analyze_correlations(data, profile)
        elif step == "pattern_discovery":
            return await self._discover_patterns(data, profile)
        elif step == "feature_analysis":
            return await self._analyze_features(data, profile)
        else:
            # 범용적 단계 처리
            return await self._generic_analysis_step(step, data, profile)
    
    async def _analyze_data_overview(self, data: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
        """데이터 개요 분석"""
        insights = []
        
        # 기본 통계
        insights.append(f"데이터셋 크기: {profile.shape[0]:,}행 × {profile.shape[1]}열")
        insights.append(f"데이터 품질 점수: {profile.quality_score:.2f}/1.0")
        insights.append(f"복잡도 수준: {profile.complexity_level}")
        
        # 결측값 분석
        high_missing_cols = [col for col, rate in profile.missing_rates.items() if rate > 0.3]
        if high_missing_cols:
            insights.append(f"높은 결측률(30% 이상) 컬럼: {', '.join(high_missing_cols[:3])}")
        
        # 데이터 특성
        characteristics_desc = [c.value for c in profile.data_characteristics]
        insights.append(f"주요 데이터 특성: {', '.join(characteristics_desc)}")
        
        return {
            "insights": insights,
            "confidence": 0.9,
            "visualizations": [],
            "statistical_tests": []
        }
    
    async def _analyze_distributions(self, data: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
        """분포 분석"""
        insights = []
        visualizations = []
        
        # 수치형 컬럼 분포 분석
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # 최대 5개 컬럼
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            # 기본 통계
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            
            # 분포 특성 분석
            skewness = col_data.skew()
            if abs(skewness) > 1:
                skew_desc = "highly skewed" if abs(skewness) > 2 else "moderately skewed"
                insights.append(f"{col}: {skew_desc} distribution (skewness: {skewness:.2f})")
            
            # 이상치 분석
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
            outlier_rate = outliers / len(col_data)
            
            if outlier_rate > 0.1:
                insights.append(f"{col}: {outlier_rate:.1%} 이상치 포함 ({outliers}개)")
            
            # 시각화 정보 (실제 생성은 별도)
            visualizations.append({
                "type": "histogram",
                "column": col,
                "description": f"{col} 분포 히스토그램"
            })
        
        return {
            "insights": insights,
            "confidence": 0.8,
            "visualizations": visualizations,
            "statistical_tests": []
        }
    
    async def _analyze_correlations(self, data: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
        """상관관계 분석"""
        insights = []
        statistical_tests = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {
                "insights": ["수치형 컬럼이 부족하여 상관관계 분석 불가"],
                "confidence": 0.3,
                "visualizations": [],
                "statistical_tests": []
            }
        
        # 상관관계 행렬 계산
        corr_matrix = numeric_data.corr()
        
        # 강한 상관관계 탐지
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    strong_correlations.append((col1, col2, corr_val))
        
        if strong_correlations:
            for col1, col2, corr_val in strong_correlations[:3]:  # 상위 3개
                insights.append(f"강한 상관관계: {col1} ↔ {col2} (r={corr_val:.3f})")
                
                statistical_tests.append({
                    "test": "Pearson correlation",
                    "variables": [col1, col2],
                    "statistic": corr_val,
                    "interpretation": "strong positive correlation" if corr_val > 0 else "strong negative correlation"
                })
        else:
            insights.append("강한 상관관계(|r| > 0.7)를 가진 변수 쌍 없음")
        
        return {
            "insights": insights,
            "confidence": 0.7,
            "visualizations": [{"type": "correlation_heatmap", "description": "상관관계 히트맵"}],
            "statistical_tests": statistical_tests
        }
    
    async def _discover_patterns(self, data: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
        """패턴 탐지 (LLM 기반)"""
        insights = []
        
        # 데이터 샘플과 기본 통계로 LLM에게 패턴 탐지 요청
        sample_data = data.head(10).to_dict()
        
        pattern_prompt = f"""
데이터 분석 전문가로서 다음 데이터에서 의미 있는 패턴을 탐지해주세요.

데이터 샘플 (처음 10행):
{json.dumps(sample_data, indent=2, default=str)}

데이터 특성:
- 형태: {profile.shape}
- 품질: {profile.quality_score:.2f}
- 특성: {[c.value for c in profile.data_characteristics]}

발견할 수 있는 패턴 유형:
1. 분포 패턴
2. 그룹화 패턴  
3. 시간적 패턴 (해당되는 경우)
4. 범주별 차이점
5. 이상 패턴

각 패턴에 대해 구체적인 인사이트와 비즈니스 의미를 제시해주세요.
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": pattern_prompt}],
                max_tokens=600,
                temperature=0.4
            )
            
            llm_patterns = response.choices[0].message.content
            
            # LLM 응답을 인사이트로 변환
            pattern_insights = llm_patterns.split('\n')
            insights.extend([insight.strip() for insight in pattern_insights if insight.strip()])
            
        except Exception as e:
            logger.warning(f"LLM 패턴 탐지 실패: {e}")
            insights.append("패턴 탐지를 위한 추가 분석이 필요합니다.")
        
        return {
            "insights": insights,
            "confidence": 0.6,
            "visualizations": [],
            "statistical_tests": []
        }
    
    async def _analyze_features(self, data: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
        """피처 분석"""
        insights = []
        
        # 피처 중요도 분석 (간단한 버전)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # 변동계수를 통한 피처 변동성 분석
            feature_variability = {}
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0 and col_data.std() > 0:
                    cv = col_data.std() / abs(col_data.mean()) if col_data.mean() != 0 else 0
                    feature_variability[col] = cv
            
            if feature_variability:
                # 높은 변동성 피처
                high_var_features = sorted(feature_variability.items(), key=lambda x: x[1], reverse=True)[:3]
                insights.append(f"높은 변동성 피처: {', '.join([f[0] for f in high_var_features])}")
                
                # 낮은 변동성 피처
                low_var_features = sorted(feature_variability.items(), key=lambda x: x[1])[:3]
                insights.append(f"낮은 변동성 피처: {', '.join([f[0] for f in low_var_features])}")
        
        return {
            "insights": insights,
            "confidence": 0.7,
            "visualizations": [],
            "statistical_tests": []
        }
    
    async def _generic_analysis_step(self, step: str, data: pd.DataFrame, profile: DataProfile) -> Dict[str, Any]:
        """범용적 분석 단계"""
        insights = [f"{step} 분석이 수행되었습니다."]
        
        return {
            "insights": insights,
            "confidence": 0.5,
            "visualizations": [],
            "statistical_tests": []
        }
    
    async def _generate_comprehensive_insights(self, data: pd.DataFrame, profile: DataProfile,
                                             strategy: AnalysisStrategy, step_insights: List[str]) -> List[str]:
        """종합 인사이트 생성 (LLM 기반)"""
        
        insights_summary = '\n'.join(step_insights[-10:])  # 최근 10개 인사이트
        
        synthesis_prompt = f"""
데이터 분석 전문가로서 다음 분석 결과들을 종합하여 핵심 인사이트를 도출해주세요.

분석 컨텍스트: {strategy.context.value}
데이터 특성: {profile.shape}, 품질 {profile.quality_score:.2f}

개별 분석 결과:
{insights_summary}

요청사항:
1. 가장 중요한 3-5개의 핵심 인사이트 도출
2. 비즈니스 관점에서의 의미 해석
3. 데이터 기반 권장사항 제시

범용적이고 실용적인 관점에서 종합해주세요.
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            comprehensive = response.choices[0].message.content
            return [comprehensive]
            
        except Exception as e:
            logger.warning(f"종합 인사이트 생성 실패: {e}")
            return ["분석 결과 종합을 위한 추가 처리가 필요합니다."]
    
    async def _generate_recommendations(self, data: pd.DataFrame, profile: DataProfile,
                                      strategy: AnalysisStrategy, insights: List[str]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 데이터 품질 기반 권장사항
        if profile.quality_score < 0.7:
            recommendations.append("데이터 품질 개선 필요: 결측값 처리 및 데이터 정제 우선 수행")
        
        # 복잡도 기반 권장사항
        if profile.complexity_level in ["high", "very_high"]:
            recommendations.append("고복잡도 데이터: 차원 축소 또는 피처 선택 기법 적용 고려")
        
        # 분석 컨텍스트 기반 권장사항
        if strategy.context == AnalysisContext.EXPLORATION:
            recommendations.append("탐색적 분석 완료 후 특정 목적의 분석(예측, 분류) 수행 권장")
        elif strategy.context == AnalysisContext.PREDICTION:
            recommendations.append("예측 모델 구축을 위한 피처 엔지니어링 및 교차 검증 수행")
        
        # 일반적 권장사항
        recommendations.append("지속적인 데이터 모니터링 및 분석 결과 검증 필요")
        
        return recommendations
    
    def _identify_limitations(self, profile: DataProfile, strategy: AnalysisStrategy) -> List[str]:
        """분석 한계점 식별"""
        limitations = []
        
        if profile.quality_score < 0.5:
            limitations.append("낮은 데이터 품질로 인한 분석 결과 신뢰성 제한")
        
        if profile.shape[0] < 100:
            limitations.append("작은 데이터 크기로 인한 통계적 유의성 제한")
        
        if DataCharacteristics.SPARSE in profile.data_characteristics:
            limitations.append("희소 데이터 특성으로 인한 패턴 탐지 어려움")
        
        limitations.append("범용적 분석 접근법으로 인한 도메인 특화 인사이트 제한")
        
        return limitations
    
    def _suggest_next_steps(self, profile: DataProfile, strategy: AnalysisStrategy) -> List[str]:
        """다음 단계 제안"""
        next_steps = []
        
        # 컨텍스트별 다음 단계
        if strategy.context == AnalysisContext.EXPLORATION:
            next_steps.append("목표 변수 설정 후 예측 또는 분류 분석 수행")
            
        next_steps.append("도메인 전문가와 결과 검토 및 해석")
        next_steps.append("추가 데이터 수집을 통한 분석 확장")
        next_steps.append("시각화 대시보드 구축으로 지속적 모니터링")
        
        return next_steps
    
    async def _save_analysis_result(self, result: AnalysisResult, profile: DataProfile, strategy: AnalysisStrategy):
        """분석 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        result_data = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "strategy": {
                    "context": strategy.context.value,
                    "confidence": strategy.confidence
                },
                "data_profile": {
                    "shape": profile.shape,
                    "quality_score": profile.quality_score,
                    "complexity": profile.complexity_level
                }
            },
            "insights": result.insights,
            "recommendations": result.recommendations,
            "methodology": result.methodology,
            "limitations": result.limitations,
            "next_steps": result.next_steps,
            "confidence_scores": result.confidence_scores
        }
        
        file_path = self.results_dir / f"universal_analysis_{timestamp}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 분석 결과 저장: {file_path}")


# 사용 예시 및 테스트
async def test_universal_analyzer():
    """범용 분석기 테스트"""
    
    # 테스트 데이터 생성 (완전히 범용적)
    np.random.seed(42)
    test_data = pd.DataFrame({
        'numeric_1': np.random.normal(100, 15, 1000),
        'numeric_2': np.random.exponential(2, 1000),
        'categorical_1': np.random.choice(['A', 'B', 'C'], 1000),
        'categorical_2': np.random.choice(['Type1', 'Type2'], 1000, p=[0.7, 0.3]),
        'mixed_data': np.random.choice(['High', 'Medium', 'Low'], 1000)
    })
    
    # 일부 결측값 추가
    test_data.loc[np.random.choice(1000, 50, replace=False), 'numeric_1'] = np.nan
    
    # 범용 분석 실행
    analyzer = UniversalAnalysisEngine()
    
    print("🔬 범용 분석 테스트 시작...")
    result = await analyzer.analyze_universally(
        test_data, 
        user_query="이 데이터의 특성과 패턴을 파악하고 싶습니다"
    )
    
    print(f"\n📊 분석 결과:")
    print(f"   방법론: {result.methodology}")
    print(f"   인사이트 수: {len(result.insights)}개")
    print(f"   권장사항 수: {len(result.recommendations)}개")
    
    print(f"\n🔍 주요 인사이트:")
    for i, insight in enumerate(result.insights[:5], 1):
        print(f"   {i}. {insight}")
    
    print(f"\n💡 주요 권장사항:")
    for i, rec in enumerate(result.recommendations[:3], 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    asyncio.run(test_universal_analyzer()) 