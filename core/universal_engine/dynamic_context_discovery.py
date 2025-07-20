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