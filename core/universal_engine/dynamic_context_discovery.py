"""
Dynamic Context Discovery - 동적 컨텍스트 발견 시스템

요구사항 3에 따른 구현:
- 데이터 특성, 패턴, 용어를 통한 도메인 컨텍스트 자동 발견
- 불확실성 처리 및 명확화 질문 생성
- 실시간 상호작용을 통한 도메인 패턴 학습
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class DynamicContextDiscovery:
    """
    동적 컨텍스트 발견 시스템
    - 데이터로부터 도메인과 요구사항 자동 발견
    - 패턴 인식 및 용어 분석
    - 점진적 컨텍스트 구축
    """
    
    def __init__(self):
        """DynamicContextDiscovery 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.discovered_contexts = {}
        self.terminology_patterns = {}
        logger.info("DynamicContextDiscovery initialized")
    
    async def discover_context(self, data: Any, query: str = None) -> Dict:
        """
        데이터와 쿼리로부터 컨텍스트 발견
        
        Args:
            data: 분석 대상 데이터
            query: 사용자 쿼리 (옵션)
            
        Returns:
            발견된 컨텍스트 정보
        """
        logger.info("Starting dynamic context discovery")
        
        try:
            # 1. 데이터 특성 분석
            data_characteristics = await self._analyze_data_characteristics(data)
            
            # 2. 패턴 및 용어 추출
            patterns_and_terms = await self._extract_patterns_and_terminology(data, data_characteristics)
            
            # 3. 도메인 컨텍스트 추론
            domain_context = await self._infer_domain_context(
                data_characteristics, patterns_and_terms, query
            )
            
            # 4. 불확실성 평가 및 명확화 필요성 판단
            uncertainty_assessment = await self._assess_uncertainty(domain_context)
            
            # 5. 관련 방법론 및 모범 사례 식별
            methodologies = await self._identify_relevant_methodologies(
                domain_context, data_characteristics
            )
            
            return {
                'data_characteristics': data_characteristics,
                'patterns_and_terms': patterns_and_terms,
                'domain_context': domain_context,
                'uncertainty_assessment': uncertainty_assessment,
                'relevant_methodologies': methodologies,
                'discovery_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in context discovery: {e}")
            raise
    
    async def _analyze_data_characteristics(self, data: Any) -> Dict:
        """
        데이터 특성 분석
        """
        characteristics = {
            'data_type': type(data).__name__,
            'structure': {},
            'statistical_properties': {},
            'patterns': {}
        }
        
        # DataFrame 분석
        if isinstance(data, pd.DataFrame):
            characteristics['structure'] = {
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'missing_values': data.isnull().sum().to_dict()
            }
            
            # 통계적 특성
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                characteristics['statistical_properties'] = {
                    'numeric_columns': list(numeric_cols),
                    'basic_stats': data[numeric_cols].describe().to_dict()
                }
            
            # 패턴 감지를 위한 샘플
            characteristics['sample_data'] = data.head(10).to_dict()
            
        # 리스트/배열 분석
        elif isinstance(data, (list, np.ndarray)):
            characteristics['structure'] = {
                'length': len(data),
                'dimensions': data.shape if hasattr(data, 'shape') else None,
                'sample': data[:5] if len(data) > 5 else data
            }
            
        # 딕셔너리 분석
        elif isinstance(data, dict):
            characteristics['structure'] = {
                'keys': list(data.keys()),
                'nested_structure': self._analyze_dict_structure(data)
            }
        
        # LLM을 통한 추가 특성 분석
        llm_analysis = await self._llm_analyze_characteristics(characteristics)
        characteristics['llm_insights'] = llm_analysis
        
        return characteristics
    
    def _analyze_dict_structure(self, d: dict, max_depth: int = 3, current_depth: int = 0) -> Dict:
        """딕셔너리 구조 분석"""
        if current_depth >= max_depth:
            return {'max_depth_reached': True}
            
        structure = {}
        for key, value in d.items():
            if isinstance(value, dict):
                structure[key] = self._analyze_dict_structure(value, max_depth, current_depth + 1)
            else:
                structure[key] = type(value).__name__
                
        return structure
    
    async def _llm_analyze_characteristics(self, characteristics: Dict) -> Dict:
        """LLM을 통한 데이터 특성 분석"""
        prompt = f"""
        다음 데이터 특성을 분석하고 도메인과 용도를 추론하세요.
        
        데이터 특성: {characteristics}
        
        친근하고 직관적인 설명으로 답하세요.
        예시: "이 데이터를 보니 뭔가 공장에서 제품을 만드는 과정을 기록한 것 같네요"
        
        JSON 형식으로 응답하세요:
        {{
            "intuitive_description": "직관적인 설명",
            "potential_domain": "추정되는 도메인",
            "data_purpose": "데이터의 목적",
            "key_observations": ["관찰1", "관찰2"],
            "confidence": "high|medium|low"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _extract_patterns_and_terminology(self, data: Any, characteristics: Dict) -> Dict:
        """
        패턴 및 용어 추출
        """
        patterns_and_terms = {
            'column_patterns': {},
            'value_patterns': {},
            'terminology': {},
            'relationships': {}
        }
        
        if isinstance(data, pd.DataFrame):
            # 컬럼명 패턴 분석
            patterns_and_terms['column_patterns'] = await self._analyze_column_patterns(data.columns)
            
            # 값 패턴 분석
            for col in data.columns[:10]:  # 처음 10개 컬럼만 분석
                if data[col].dtype == object:
                    patterns_and_terms['value_patterns'][col] = await self._analyze_value_patterns(
                        data[col].dropna().unique()[:20]
                    )
            
            # 관계 분석
            patterns_and_terms['relationships'] = await self._analyze_relationships(data)
        
        # LLM을 통한 용어 및 패턴 해석
        terminology_analysis = await self._interpret_terminology(patterns_and_terms)
        patterns_and_terms['terminology_interpretation'] = terminology_analysis
        
        return patterns_and_terms
    
    async def _analyze_column_patterns(self, columns: List[str]) -> Dict:
        """컬럼명 패턴 분석"""
        prompt = f"""
        다음 컬럼명들을 분석하여 패턴과 도메인 특성을 파악하세요.
        
        컬럼명: {list(columns)}
        
        JSON 형식으로 응답하세요:
        {{
            "naming_convention": "스네이크케이스|캐멀케이스|기타",
            "language": "영어|한국어|혼합|기타",
            "domain_indicators": ["도메인 지시자1", "도메인 지시자2"],
            "grouped_columns": {{
                "그룹1": ["컬럼1", "컬럼2"],
                "그룹2": ["컬럼3", "컬럼4"]
            }},
            "technical_terms": ["기술 용어1", "기술 용어2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _analyze_value_patterns(self, values: np.ndarray) -> Dict:
        """값 패턴 분석"""
        if len(values) == 0:
            return {}
            
        sample_values = list(values[:10])
        
        prompt = f"""
        다음 값들의 패턴을 분석하세요.
        
        샘플 값: {sample_values}
        
        JSON 형식으로 응답하세요:
        {{
            "value_type": "범주형|숫자형|날짜|텍스트|코드|기타",
            "pattern_description": "패턴 설명",
            "potential_meaning": "잠재적 의미",
            "domain_specific": true/false
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _analyze_relationships(self, data: pd.DataFrame) -> Dict:
        """데이터 내 관계 분석"""
        relationships = {
            'correlations': {},
            'dependencies': {},
            'hierarchies': {}
        }
        
        # 숫자형 컬럼 간 상관관계
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            high_corr = []
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'col1': numeric_cols[i],
                            'col2': numeric_cols[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            relationships['correlations'] = high_corr
        
        return relationships
    
    async def _interpret_terminology(self, patterns_and_terms: Dict) -> Dict:
        """용어 해석"""
        prompt = f"""
        다음 패턴과 용어를 분석하여 도메인과 전문 용어를 해석하세요.
        
        패턴과 용어: {patterns_and_terms}
        
        JSON 형식으로 응답하세요:
        {{
            "identified_domain": "식별된 도메인",
            "domain_confidence": "high|medium|low",
            "key_terminology": {{
                "용어1": "설명1",
                "용어2": "설명2"
            }},
            "domain_specific_patterns": ["패턴1", "패턴2"],
            "suggested_analysis_focus": ["분석 초점1", "분석 초점2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _infer_domain_context(self, characteristics: Dict, patterns: Dict, query: str = None) -> Dict:
        """
        도메인 컨텍스트 추론
        """
        prompt = f"""
        데이터 특성과 패턴을 바탕으로 도메인 컨텍스트를 추론하세요.
        
        데이터 특성: {characteristics.get('llm_insights', {})}
        패턴 분석: {patterns.get('terminology_interpretation', {})}
        사용자 쿼리: {query if query else "없음"}
        
        하드코딩된 도메인 카테고리를 사용하지 말고,
        데이터 자체에서 발견한 것을 바탕으로 추론하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "domain": "추론된 도메인",
            "sub_domain": "세부 도메인",
            "confidence_level": 0.0-1.0,
            "evidence": ["근거1", "근거2", "근거3"],
            "domain_characteristics": {{
                "key_concepts": ["핵심 개념1", "핵심 개념2"],
                "typical_analyses": ["일반적 분석1", "일반적 분석2"],
                "important_metrics": ["중요 지표1", "중요 지표2"]
            }},
            "user_intent_alignment": "사용자 의도와의 정렬도"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _assess_uncertainty(self, domain_context: Dict) -> Dict:
        """
        불확실성 평가 및 명확화 필요성 판단
        """
        confidence = domain_context.get('confidence_level', 0.5)
        
        uncertainty_assessment = {
            'overall_confidence': confidence,
            'uncertain_areas': [],
            'clarification_needed': [],
            'confidence_breakdown': {}
        }
        
        # 낮은 신뢰도 영역 식별
        if confidence < 0.7:
            prompt = f"""
            도메인 컨텍스트의 불확실한 부분을 식별하고 명확화가 필요한 질문을 생성하세요.
            
            도메인 컨텍스트: {domain_context}
            
            JSON 형식으로 응답하세요:
            {{
                "uncertain_areas": [
                    {{
                        "area": "불확실한 영역",
                        "reason": "불확실한 이유",
                        "impact": "분석에 미치는 영향"
                    }}
                ],
                "clarification_questions": [
                    {{
                        "question": "명확화 질문",
                        "purpose": "질문의 목적",
                        "options": ["선택지1", "선택지2", "기타"]
                    }}
                ],
                "alternative_interpretations": [
                    {{
                        "interpretation": "대안 해석",
                        "likelihood": "high|medium|low"
                    }}
                ]
            }}
            """
            
            response = await self.llm_client.agenerate(prompt)
            uncertainty_details = self._parse_json_response(response)
            
            uncertainty_assessment.update(uncertainty_details)
        
        return uncertainty_assessment
    
    async def _identify_relevant_methodologies(self, domain_context: Dict, characteristics: Dict) -> Dict:
        """
        관련 방법론 및 모범 사례 식별
        """
        prompt = f"""
        도메인 컨텍스트와 데이터 특성을 바탕으로 적절한 분석 방법론을 제안하세요.
        
        도메인: {domain_context.get('domain', '알 수 없음')}
        데이터 특성: {characteristics.get('structure', {})}
        도메인 특성: {domain_context.get('domain_characteristics', {})}
        
        하드코딩된 방법론이 아닌, 데이터와 도메인에 맞는 방법론을 동적으로 생성하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "recommended_methodologies": [
                {{
                    "method": "방법론 이름",
                    "description": "설명",
                    "applicability": "이 상황에 적합한 이유",
                    "expected_insights": ["예상 인사이트1", "예상 인사이트2"]
                }}
            ],
            "best_practices": [
                {{
                    "practice": "모범 사례",
                    "rationale": "이유",
                    "implementation": "구현 방법"
                }}
            ],
            "analysis_workflow": [
                {{
                    "step": 1,
                    "action": "수행할 작업",
                    "purpose": "목적"
                }}
            ],
            "potential_pitfalls": ["주의사항1", "주의사항2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def refine_context_with_feedback(self, current_context: Dict, user_feedback: str) -> Dict:
        """
        사용자 피드백을 통한 컨텍스트 개선
        """
        prompt = f"""
        사용자 피드백을 바탕으로 도메인 컨텍스트를 개선하세요.
        
        현재 컨텍스트: {current_context}
        사용자 피드백: {user_feedback}
        
        JSON 형식으로 개선된 컨텍스트를 제공하세요:
        {{
            "refined_domain": "개선된 도메인 이해",
            "corrections": ["수정사항1", "수정사항2"],
            "new_insights": ["새로운 인사이트1", "새로운 인사이트2"],
            "updated_confidence": 0.0-1.0,
            "learning_notes": "이번 상호작용에서 배운 점"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        refined = self._parse_json_response(response)
        
        # 개선된 컨텍스트를 기존 컨텍스트와 병합
        updated_context = {**current_context}
        updated_context['refinements'] = refined
        updated_context['confidence_level'] = refined.get('updated_confidence', current_context.get('confidence_level', 0.5))
        
        return updated_context
    
    async def analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """
        데이터 특성 자동 분석
        
        요구사항 1.1에 따른 구현:
        - 데이터 타입별 분석 로직 (tabular, sequence, dictionary)
        - 패턴 감지 및 품질 평가 시스템
        - LLM 기반 동적 특성 분석
        
        Args:
            data: 분석할 데이터
            
        Returns:
            데이터 특성 분석 결과
        """
        logger.info("Analyzing data characteristics with dynamic discovery")
        
        try:
            # 기본 특성 분석 수행
            basic_characteristics = await self._analyze_data_characteristics(data)
            
            # 심화 패턴 분석
            pattern_analysis = await self._perform_deep_pattern_analysis(data, basic_characteristics)
            
            # 품질 평가
            quality_assessment = await self._assess_data_quality(data, basic_characteristics)
            
            # 도메인 힌트 추출
            domain_hints = await self._extract_domain_hints(data, basic_characteristics, pattern_analysis)
            
            # 분석 가능성 평가
            analysis_potential = await self._evaluate_analysis_potential(
                data, basic_characteristics, pattern_analysis
            )
            
            # 통합 결과 생성
            comprehensive_analysis = {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_overview': {
                    'type': basic_characteristics.get('data_type'),
                    'size': basic_characteristics.get('structure', {}).get('rows', 'unknown'),
                    'complexity': self._calculate_data_complexity(basic_characteristics)
                },
                'structural_analysis': basic_characteristics.get('structure', {}),
                'statistical_properties': basic_characteristics.get('statistical_properties', {}),
                'pattern_analysis': pattern_analysis,
                'quality_assessment': quality_assessment,
                'domain_hints': domain_hints,
                'analysis_potential': analysis_potential,
                'recommendations': await self._generate_analysis_recommendations(
                    basic_characteristics, pattern_analysis, quality_assessment
                ),
                'confidence_score': self._calculate_analysis_confidence(
                    basic_characteristics, pattern_analysis, quality_assessment
                )
            }
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in data characteristics analysis: {e}")
            return {
                'error': str(e),
                'fallback_analysis': await self._fallback_data_analysis(data),
                'timestamp': datetime.now().isoformat()
            }
    
    async def detect_domain(self, data: Any, query: str) -> Dict[str, Any]:
        """
        도메인 컨텍스트 자동 감지
        
        요구사항 1.1에 따른 구현:
        - LLM 기반 도메인 컨텍스트 감지
        - 데이터와 쿼리의 종합적 분석
        - 불확실성 처리 및 명확화 질문 생성
        
        Args:
            data: 분석 대상 데이터
            query: 사용자 쿼리
            
        Returns:
            도메인 감지 결과
        """
        logger.info("Detecting domain context dynamically")
        
        try:
            # 데이터 특성 분석
            data_characteristics = await self.analyze_data_characteristics(data)
            
            # 쿼리 의도 분석
            query_analysis = await self._analyze_query_intent(query)
            
            # 도메인 감지 수행
            domain_detection = await self._perform_domain_detection(
                data_characteristics, query_analysis, data, query
            )
            
            # 컨텍스트 신뢰도 평가
            confidence_assessment = await self._assess_domain_confidence(
                domain_detection, data_characteristics, query_analysis
            )
            
            # 명확화 필요성 판단
            clarification_needs = await self._determine_clarification_needs(
                domain_detection, confidence_assessment
            )
            
            # 도메인별 분석 전략 제안
            analysis_strategy = await self._suggest_domain_analysis_strategy(
                domain_detection, data_characteristics
            )
            
            # 통합 결과 생성
            domain_context = {
                'detection_timestamp': datetime.now().isoformat(),
                'query': query,
                'detected_domain': {
                    'primary_domain': domain_detection.get('domain', 'unknown'),
                    'sub_domains': domain_detection.get('sub_domains', []),
                    'domain_confidence': confidence_assessment.get('overall_confidence', 0.0),
                    'evidence': domain_detection.get('evidence', [])
                },
                'data_domain_alignment': {
                    'alignment_score': self._calculate_alignment_score(
                        data_characteristics, domain_detection
                    ),
                    'supporting_features': domain_detection.get('supporting_features', []),
                    'conflicting_features': domain_detection.get('conflicting_features', [])
                },
                'query_domain_alignment': {
                    'intent_clarity': query_analysis.get('clarity_score', 0.0),
                    'domain_specificity': query_analysis.get('domain_specificity', 0.0),
                    'terminology_match': query_analysis.get('terminology_match', 0.0)
                },
                'confidence_assessment': confidence_assessment,
                'clarification_needs': clarification_needs,
                'analysis_strategy': analysis_strategy,
                'uncertainty_handling': {
                    'uncertain_aspects': confidence_assessment.get('uncertain_aspects', []),
                    'risk_mitigation': await self._suggest_risk_mitigation(confidence_assessment),
                    'fallback_strategies': await self._suggest_fallback_strategies(domain_detection)
                }
            }
            
            return domain_context
            
        except Exception as e:
            logger.error(f"Error in domain detection: {e}")
            return {
                'error': str(e),
                'fallback_domain': await self._fallback_domain_detection(query),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _perform_deep_pattern_analysis(self, data: Any, basic_characteristics: Dict) -> Dict:
        """심화 패턴 분석"""
        pattern_prompt = f"""
        데이터의 심화 패턴을 분석하세요.
        
        기본 특성: {basic_characteristics}
        
        다음 관점에서 분석하세요:
        1. 시간적 패턴 (시계열, 주기성, 트렌드)
        2. 공간적 패턴 (지역성, 분포, 클러스터링)
        3. 범주적 패턴 (그룹화, 계층구조, 분류)
        4. 관계적 패턴 (상관관계, 의존성, 인과관계)
        5. 이상 패턴 (아웃라이어, 예외, 특이점)
        
        JSON 형식으로 응답하세요:
        {{
            "temporal_patterns": {{
                "has_time_component": true/false,
                "periodicity": "일별|주별|월별|연별|없음",
                "trend": "증가|감소|안정|변동",
                "seasonality": "있음|없음"
            }},
            "spatial_patterns": {{
                "has_spatial_component": true/false,
                "distribution_type": "균등|집중|분산|클러스터",
                "geographic_scope": "local|regional|national|global|unknown"
            }},
            "categorical_patterns": {{
                "category_structure": "flat|hierarchical|network",
                "category_balance": "균등|불균등|극도불균등",
                "dominant_categories": ["카테고리1", "카테고리2"]
            }},
            "relational_patterns": {{
                "strong_correlations": ["관계1", "관계2"],
                "dependency_chains": ["의존성1", "의존성2"],
                "interaction_effects": ["상호작용1", "상호작용2"]
            }},
            "anomaly_patterns": {{
                "outlier_presence": "high|medium|low|none",
                "anomaly_types": ["타입1", "타입2"],
                "data_quality_issues": ["이슈1", "이슈2"]
            }}
        }}
        """
        
        response = await self.llm_client.agenerate(pattern_prompt)
        return self._parse_json_response(response)
    
    async def _assess_data_quality(self, data: Any, characteristics: Dict) -> Dict:
        """데이터 품질 평가"""
        quality_prompt = f"""
        데이터 품질을 종합적으로 평가하세요.
        
        데이터 특성: {characteristics}
        
        평가 기준:
        1. 완전성 (결측값, 누락 데이터)
        2. 정확성 (오류, 이상값)
        3. 일관성 (형식, 단위, 표준)
        4. 적시성 (최신성, 관련성)
        5. 유효성 (범위, 제약조건)
        
        JSON 형식으로 응답하세요:
        {{
            "overall_quality_score": 0.0-1.0,
            "completeness": {{
                "score": 0.0-1.0,
                "missing_data_ratio": 0.0-1.0,
                "critical_missing": ["중요 누락1", "중요 누락2"]
            }},
            "accuracy": {{
                "score": 0.0-1.0,
                "potential_errors": ["오류1", "오류2"],
                "outlier_ratio": 0.0-1.0
            }},
            "consistency": {{
                "score": 0.0-1.0,
                "format_issues": ["형식 문제1", "형식 문제2"],
                "unit_consistency": "good|fair|poor"
            }},
            "timeliness": {{
                "score": 0.0-1.0,
                "data_freshness": "current|recent|outdated|unknown",
                "relevance": "high|medium|low"
            }},
            "validity": {{
                "score": 0.0-1.0,
                "constraint_violations": ["위반1", "위반2"],
                "range_issues": ["범위 문제1", "범위 문제2"]
            }},
            "recommendations": ["개선 권장사항1", "개선 권장사항2"]
        }}
        """
        
        response = await self.llm_client.agenerate(quality_prompt)
        return self._parse_json_response(response)
    
    async def _extract_domain_hints(self, data: Any, characteristics: Dict, patterns: Dict) -> Dict:
        """도메인 힌트 추출"""
        hints_prompt = f"""
        데이터 특성과 패턴에서 도메인 힌트를 추출하세요.
        
        특성: {characteristics.get('llm_insights', {})}
        패턴: {patterns}
        
        JSON 형식으로 응답하세요:
        {{
            "domain_indicators": [
                {{
                    "indicator": "지시자",
                    "evidence": "근거",
                    "confidence": 0.0-1.0
                }}
            ],
            "terminology_clues": ["용어 단서1", "용어 단서2"],
            "structural_clues": ["구조적 단서1", "구조적 단서2"],
            "pattern_clues": ["패턴 단서1", "패턴 단서2"],
            "likely_domains": [
                {{
                    "domain": "도메인명",
                    "probability": 0.0-1.0,
                    "reasoning": "추론 근거"
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(hints_prompt)
        return self._parse_json_response(response)
    
    async def _analyze_query_intent(self, query: str) -> Dict:
        """쿼리 의도 분석"""
        intent_prompt = f"""
        사용자 쿼리의 의도를 분석하세요.
        
        쿼리: "{query}"
        
        JSON 형식으로 응답하세요:
        {{
            "primary_intent": "분석|시각화|예측|분류|설명|비교|기타",
            "secondary_intents": ["보조 의도1", "보조 의도2"],
            "clarity_score": 0.0-1.0,
            "domain_specificity": 0.0-1.0,
            "technical_level": "beginner|intermediate|expert",
            "terminology_match": 0.0-1.0,
            "ambiguity_areas": ["모호한 부분1", "모호한 부분2"],
            "implicit_requirements": ["암시적 요구사항1", "암시적 요구사항2"]
        }}
        """
        
        response = await self.llm_client.agenerate(intent_prompt)
        return self._parse_json_response(response)
    
    async def _perform_domain_detection(self, data_characteristics: Dict, query_analysis: Dict, data: Any, query: str) -> Dict:
        """도메인 감지 수행"""
        detection_prompt = f"""
        데이터와 쿼리를 종합하여 도메인을 감지하세요.
        
        데이터 특성: {data_characteristics.get('domain_hints', {})}
        쿼리 분석: {query_analysis}
        원본 쿼리: "{query}"
        
        하드코딩된 도메인 목록을 사용하지 말고, 
        실제 데이터와 쿼리에서 발견되는 증거를 바탕으로 추론하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "domain": "감지된 주요 도메인",
            "sub_domains": ["세부 도메인1", "세부 도메인2"],
            "confidence": 0.0-1.0,
            "evidence": [
                {{
                    "type": "data|query|pattern|terminology",
                    "evidence": "구체적 근거",
                    "weight": 0.0-1.0
                }}
            ],
            "supporting_features": ["지지 특징1", "지지 특징2"],
            "conflicting_features": ["상충 특징1", "상충 특징2"],
            "alternative_domains": [
                {{
                    "domain": "대안 도메인",
                    "probability": 0.0-1.0
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(detection_prompt)
        return self._parse_json_response(response)
    
    def _calculate_data_complexity(self, characteristics: Dict) -> str:
        """데이터 복잡도 계산"""
        structure = characteristics.get('structure', {})
        
        # 간단한 복잡도 계산 로직
        if isinstance(structure.get('columns'), int):
            cols = structure['columns']
            rows = structure.get('rows', 0)
            
            if cols <= 5 and rows <= 1000:
                return 'low'
            elif cols <= 20 and rows <= 10000:
                return 'medium'
            else:
                return 'high'
        
        return 'unknown'
    
    def _calculate_analysis_confidence(self, characteristics: Dict, patterns: Dict, quality: Dict) -> float:
        """분석 신뢰도 계산"""
        confidence_factors = [
            quality.get('overall_quality_score', 0.5),
            1.0 if characteristics.get('llm_insights') else 0.5,
            0.8 if patterns else 0.3
        ]
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_alignment_score(self, data_characteristics: Dict, domain_detection: Dict) -> float:
        """데이터-도메인 정렬 점수 계산"""
        alignment_factors = [
            domain_detection.get('confidence', 0.0),
            len(domain_detection.get('supporting_features', [])) / 10.0,
            max(0, 1.0 - len(domain_detection.get('conflicting_features', [])) / 5.0)
        ]
        return min(sum(alignment_factors) / len(alignment_factors), 1.0)
    
    async def _fallback_data_analysis(self, data: Any) -> Dict:
        """폴백 데이터 분석"""
        return {
            'type': 'fallback',
            'basic_info': {
                'data_type': type(data).__name__,
                'size': len(data) if hasattr(data, '__len__') else 'unknown'
            },
            'message': '기본 데이터 분석만 가능합니다.'
        }
    
    async def _fallback_domain_detection(self, query: str) -> Dict:
        """폴백 도메인 감지"""
        return {
            'type': 'fallback',
            'domain': 'general',
            'confidence': 0.1,
            'message': f'쿼리 "{query}"에 대한 도메인을 감지할 수 없습니다.'
        }
    
    async def _evaluate_analysis_potential(self, data: Any, characteristics: Dict, patterns: Dict) -> Dict:
        """분석 가능성 평가"""
        return {
            'analysis_feasibility': 'high',
            'recommended_analyses': ['기술통계', '시각화', '패턴분석'],
            'data_readiness': 0.8,
            'potential_insights': ['데이터 분포', '주요 패턴', '이상값']
        }
    
    async def _generate_analysis_recommendations(self, characteristics: Dict, patterns: Dict, quality: Dict) -> List[str]:
        """분석 권장사항 생성"""
        recommendations = []
        
        # 품질 기반 권장사항
        if quality.get('overall_quality_score', 0.5) < 0.7:
            recommendations.append("데이터 품질 개선 필요")
        
        # 패턴 기반 권장사항
        if patterns.get('temporal_patterns', {}).get('has_time_component'):
            recommendations.append("시계열 분석 고려")
        
        recommendations.extend([
            "기본 통계 분석 수행",
            "데이터 시각화 생성",
            "패턴 및 이상값 탐지"
        ])
        
        return recommendations[:5]
    
    async def _assess_domain_confidence(self, domain_detection: Dict, data_characteristics: Dict, query_analysis: Dict) -> Dict:
        """도메인 신뢰도 평가"""
        base_confidence = domain_detection.get('confidence', 0.5)
        data_support = len(domain_detection.get('supporting_features', [])) / 10.0
        query_clarity = query_analysis.get('clarity_score', 0.5)
        
        overall_confidence = (base_confidence + data_support + query_clarity) / 3.0
        
        return {
            'overall_confidence': min(overall_confidence, 1.0),
            'confidence_factors': {
                'domain_detection': base_confidence,
                'data_support': data_support,
                'query_clarity': query_clarity
            },
            'uncertain_aspects': [] if overall_confidence > 0.7 else ['도메인 분류', '데이터 적합성']
        }
    
    async def _determine_clarification_needs(self, domain_detection: Dict, confidence_assessment: Dict) -> Dict:
        """명확화 필요성 판단"""
        confidence = confidence_assessment.get('overall_confidence', 0.5)
        
        if confidence < 0.6:
            return {
                'needs_clarification': True,
                'clarification_questions': [
                    "데이터의 출처나 수집 목적을 알려주실 수 있나요?",
                    "특별히 관심 있는 분석 영역이 있나요?",
                    "이 데이터로 해결하고자 하는 문제가 무엇인가요?"
                ],
                'priority': 'high' if confidence < 0.4 else 'medium'
            }
        else:
            return {
                'needs_clarification': False,
                'confidence_sufficient': True
            }
    
    async def _suggest_domain_analysis_strategy(self, domain_detection: Dict, data_characteristics: Dict) -> Dict:
        """도메인별 분석 전략 제안"""
        domain = domain_detection.get('domain', 'general')
        
        return {
            'primary_strategy': f"{domain} 도메인에 특화된 분석 접근법",
            'analysis_steps': [
                "데이터 탐색적 분석",
                "도메인별 핵심 지표 계산",
                "패턴 및 이상값 탐지",
                "인사이트 도출 및 해석"
            ],
            'domain_specific_considerations': [
                f"{domain} 도메인의 특성 고려",
                "업계 표준 및 모범 사례 적용"
            ]
        }
    
    async def _suggest_risk_mitigation(self, confidence_assessment: Dict) -> List[str]:
        """위험 완화 제안"""
        confidence = confidence_assessment.get('overall_confidence', 0.5)
        
        if confidence < 0.5:
            return [
                "추가 데이터 수집 고려",
                "도메인 전문가 자문 요청",
                "다중 접근법으로 검증",
                "결과 해석 시 불확실성 명시"
            ]
        else:
            return [
                "결과 검증 및 교차 확인",
                "가정 사항 명확히 문서화"
            ]
    
    async def _suggest_fallback_strategies(self, domain_detection: Dict) -> List[str]:
        """폴백 전략 제안"""
        return [
            "일반적인 데이터 분석 접근법 적용",
            "기본 통계 분석으로 시작",
            "점진적으로 도메인 특화 분석 확장",
            "사용자 피드백을 통한 방향 조정"
        ]
    
    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        import json
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {
                'raw_response': response,
                'parse_error': str(e)
            }
    
    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        import json
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {
                'raw_response': response,
                'parse_error': str(e)
            }