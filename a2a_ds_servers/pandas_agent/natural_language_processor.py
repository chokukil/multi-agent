"""
🧠 Natural Language Processor

자연어 쿼리 분석 및 처리 시스템
데이터 분석 의도 파악 및 적절한 분석 방법 매핑

Author: CherryAI Team
License: MIT License
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """쿼리 유형 분류"""
    SUMMARY = "summary"                    # 데이터 요약
    STATISTICS = "statistics"              # 기술통계
    VISUALIZATION = "visualization"        # 시각화
    FILTERING = "filtering"                # 데이터 필터링
    AGGREGATION = "aggregation"            # 집계 연산
    CORRELATION = "correlation"            # 상관관계 분석
    MISSING_DATA = "missing_data"          # 결측 데이터 분석
    COMPARISON = "comparison"              # 비교 분석
    TREND = "trend"                        # 트렌드 분석
    DISTRIBUTION = "distribution"          # 분포 분석
    GROUPBY = "groupby"                    # 그룹별 분석
    MERGE_JOIN = "merge_join"              # 데이터 병합
    TRANSFORMATION = "transformation"       # 데이터 변환
    GENERAL = "general"                    # 일반 질문


@dataclass
class QueryIntent:
    """쿼리 의도 분석 결과"""
    query_type: QueryType
    confidence: float
    keywords: List[str]
    target_columns: List[str]
    operations: List[str]
    filters: Dict[str, Any]
    aggregations: List[str]
    visualization_type: Optional[str] = None


class NaturalLanguageProcessor:
    """자연어 처리기"""
    
    def __init__(self):
        self.query_patterns = self._initialize_patterns()
        self.column_aliases = {
            # 일반적인 컬럼 별칭
            "나이": ["age", "나이", "연령"],
            "성별": ["gender", "sex", "성별"],
            "가격": ["price", "cost", "가격", "비용", "금액"],
            "날짜": ["date", "time", "날짜", "시간", "일자"],
            "이름": ["name", "이름", "명칭"],
            "수량": ["quantity", "count", "수량", "개수"],
            "상태": ["status", "state", "상태"],
            "카테고리": ["category", "type", "카테고리", "유형", "분류"]
        }
    
    def _initialize_patterns(self) -> Dict[QueryType, List[Dict]]:
        """쿼리 패턴 초기화"""
        return {
            QueryType.SUMMARY: [
                {"pattern": r"요약|개요|overview|summary|전체|살펴", "weight": 1.0},
                {"pattern": r"어떤.*데이터|무엇.*포함|뭐.*들어", "weight": 0.8},
                {"pattern": r"보여줘|알려줘|설명", "weight": 0.6}
            ],
            QueryType.STATISTICS: [
                {"pattern": r"통계|기술통계|describe|평균|mean|중앙값|median", "weight": 1.0},
                {"pattern": r"최대|최소|max|min|표준편차|std|분산|variance", "weight": 0.9},
                {"pattern": r"분포|distribution|히스토그램|histogram", "weight": 0.8}
            ],
            QueryType.VISUALIZATION: [
                {"pattern": r"그래프|차트|plot|그림|시각화|visualization", "weight": 1.0},
                {"pattern": r"그려|그린|플롯|막대|선|원|scatter|bar|line|pie", "weight": 0.9},
                {"pattern": r"보여줘.*그래프|차트.*생성", "weight": 0.8}
            ],
            QueryType.FILTERING: [
                {"pattern": r"필터|filter|조건|condition|where", "weight": 1.0},
                {"pattern": r"~인|~가 있는|~보다 큰|~보다 작은|~와 같은", "weight": 0.9},
                {"pattern": r"포함.*데이터|해당.*행|특정.*값", "weight": 0.7}
            ],
            QueryType.AGGREGATION: [
                {"pattern": r"합계|sum|총합|전체.*더한|총.*개수", "weight": 1.0},
                {"pattern": r"평균|mean|average|중앙값|median", "weight": 0.9},
                {"pattern": r"최대|최소|max|min|개수|count", "weight": 0.8}
            ],
            QueryType.CORRELATION: [
                {"pattern": r"상관관계|correlation|관계|연관|영향", "weight": 1.0},
                {"pattern": r"~와.*관련|~에.*따른|~와.*비례", "weight": 0.9},
                {"pattern": r"관련성|상관성|연관성", "weight": 0.8}
            ],
            QueryType.MISSING_DATA: [
                {"pattern": r"결측|missing|null|nan|빈.*값|없는.*데이터", "weight": 1.0},
                {"pattern": r"누락|비어있는|공백|empty", "weight": 0.9}
            ],
            QueryType.COMPARISON: [
                {"pattern": r"비교|compare|차이|difference|대비", "weight": 1.0},
                {"pattern": r"~보다|~와.*다른|~에.*비해", "weight": 0.9},
                {"pattern": r"높은|낮은|많은|적은", "weight": 0.7}
            ],
            QueryType.TREND: [
                {"pattern": r"트렌드|trend|추세|변화|시간.*따른", "weight": 1.0},
                {"pattern": r"증가|감소|상승|하락|변동", "weight": 0.9},
                {"pattern": r"시계열|time.*series|월별|년도별", "weight": 0.8}
            ],
            QueryType.GROUPBY: [
                {"pattern": r"그룹|group|별로|~마다|~당", "weight": 1.0},
                {"pattern": r"카테고리.*별|지역.*별|성별.*별", "weight": 0.9},
                {"pattern": r"분류.*해서|나누어서", "weight": 0.8}
            ],
            QueryType.MERGE_JOIN: [
                {"pattern": r"합치|merge|join|병합|결합", "weight": 1.0},
                {"pattern": r"연결|합쳐서|함께.*분석", "weight": 0.9}
            ]
        }
    
    async def analyze_query(self, query: str, available_columns: List[str] = None) -> QueryIntent:
        """쿼리 의도 분석"""
        try:
            query_lower = query.lower().strip()
            
            # 1. 쿼리 유형 분류
            query_type, confidence = self._classify_query_type(query_lower)
            
            # 2. 키워드 추출
            keywords = self._extract_keywords(query_lower)
            
            # 3. 대상 컬럼 추출
            target_columns = self._extract_target_columns(query, available_columns or [])
            
            # 4. 연산 추출
            operations = self._extract_operations(query_lower)
            
            # 5. 필터 조건 추출
            filters = self._extract_filters(query_lower)
            
            # 6. 집계 연산 추출
            aggregations = self._extract_aggregations(query_lower)
            
            # 7. 시각화 유형 추출
            visualization_type = self._extract_visualization_type(query_lower) if query_type == QueryType.VISUALIZATION else None
            
            intent = QueryIntent(
                query_type=query_type,
                confidence=confidence,
                keywords=keywords,
                target_columns=target_columns,
                operations=operations,
                filters=filters,
                aggregations=aggregations,
                visualization_type=visualization_type
            )
            
            logger.info(f"🧠 쿼리 분석 완료: {query_type.value} (신뢰도: {confidence:.2f})")
            return intent
            
        except Exception as e:
            logger.error(f"❌ 쿼리 분석 실패: {e}")
            # 기본 의도 반환
            return QueryIntent(
                query_type=QueryType.GENERAL,
                confidence=0.5,
                keywords=[],
                target_columns=[],
                operations=[],
                filters={},
                aggregations=[]
            )
    
    def _classify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """쿼리 유형 분류"""
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0.0
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                weight = pattern_info["weight"]
                
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches * weight
            
            if score > 0:
                scores[query_type] = score
        
        if not scores:
            return QueryType.GENERAL, 0.5
        
        # 최고 점수 쿼리 유형 반환
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        # 신뢰도 계산 (0.5 ~ 1.0 범위로 정규화)
        confidence = min(0.5 + (max_score * 0.1), 1.0)
        
        return best_type, confidence
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출 - 한국어 어미 처리 개선"""
        # 불용어 제거
        stop_words = {
            "의", "를", "을", "이", "가", "은", "는", "에", "로", "으로", "와", "과", 
            "해줘", "보여줘", "알려줘", "주세요", "해주세요", "입니다", "습니다", "에서", "부터", "까지"
        }
        
        # 1. 패턴 기반 키워드 추출
        pattern_keywords = []
        for query_type, patterns in self.query_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                matches = re.findall(pattern, query, re.IGNORECASE)
                pattern_keywords.extend(matches)
        
        # 2. 단어 분리 및 정제
        words = re.findall(r'[가-힣\w]+', query)
        
        # 3. 한국어 어미 제거
        processed_words = []
        for word in words:
            if len(word) > 1 and word not in stop_words:
                # 한국어 어미 제거 (요약을 -> 요약, 분석해 -> 분석)
                stem = re.sub(r'[을를은는이가에서의와과도만까지부터야라]$', '', word)
                stem = re.sub(r'[해하게려고려면하면한다고네요]$', '', stem)
                processed_words.append(stem if len(stem) > 1 else word)
        
        # 4. 모든 키워드 합치기
        all_keywords = pattern_keywords + processed_words
        
        # 중복 제거 및 빈 문자열 제거
        keywords = list(set(word for word in all_keywords if word and len(word) > 1))
        
        return keywords[:10]  # 상위 10개만 반환
    
    def _extract_target_columns(self, query: str, available_columns: List[str]) -> List[str]:
        """대상 컬럼 추출"""
        target_columns = []
        query_lower = query.lower()
        
        # 1. 직접 컬럼명 매칭
        for col in available_columns:
            if col.lower() in query_lower:
                target_columns.append(col)
        
        # 2. 별칭을 통한 매칭
        for korean_name, aliases in self.column_aliases.items():
            for alias in aliases:
                if alias in query_lower:
                    # 해당 별칭과 매칭되는 실제 컬럼 찾기
                    matching_cols = [col for col in available_columns 
                                   if any(a.lower() in col.lower() for a in aliases)]
                    target_columns.extend(matching_cols)
        
        # 중복 제거
        return list(set(target_columns))
    
    def _extract_operations(self, query: str) -> List[str]:
        """연산 추출"""
        operations = []
        
        operation_patterns = {
            "count": r"개수|count|수|갯수",
            "sum": r"합계|sum|총합|더한",
            "mean": r"평균|mean|average",
            "median": r"중앙값|median",
            "max": r"최대|최고|max|가장.*큰",
            "min": r"최소|최저|min|가장.*작은",
            "std": r"표준편차|std|standard",
            "var": r"분산|variance|var",
            "unique": r"고유|unique|유일|distinct",
            "sort": r"정렬|sort|순서"
        }
        
        for op, pattern in operation_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                operations.append(op)
        
        return operations
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """필터 조건 추출"""
        filters = {}
        
        # 숫자 필터 패턴
        number_patterns = [
            (r"(\w+).*보다.*큰.*?(\d+(?:\.\d+)?)", "gt"),
            (r"(\w+).*보다.*작은.*?(\d+(?:\.\d+)?)", "lt"),
            (r"(\w+).*이상.*?(\d+(?:\.\d+)?)", "gte"),
            (r"(\w+).*이하.*?(\d+(?:\.\d+)?)", "lte"),
            (r"(\w+).*같은.*?(\d+(?:\.\d+)?)", "eq")
        ]
        
        for pattern, operator in number_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for column, value in matches:
                if column not in filters:
                    filters[column] = []
                filters[column].append({"operator": operator, "value": float(value)})
        
        # 텍스트 필터 패턴
        text_patterns = [
            (r"(\w+).*포함.*['\"](.+?)['\"]", "contains"),
            (r"(\w+).*같은.*['\"](.+?)['\"]", "equals"),
            (r"(\w+).*시작.*['\"](.+?)['\"]", "startswith")
        ]
        
        for pattern, operator in text_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for column, value in matches:
                if column not in filters:
                    filters[column] = []
                filters[column].append({"operator": operator, "value": value})
        
        return filters
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """집계 연산 추출"""
        aggregations = []
        
        agg_patterns = {
            "sum": r"합계|sum|총합",
            "mean": r"평균|mean|average",
            "count": r"개수|count|갯수",
            "median": r"중앙값|median",
            "max": r"최대|최고|max",
            "min": r"최소|최저|min",
            "std": r"표준편차|std",
            "var": r"분산|variance"
        }
        
        for agg, pattern in agg_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                aggregations.append(agg)
        
        return aggregations
    
    def _extract_visualization_type(self, query: str) -> Optional[str]:
        """시각화 유형 추출"""
        viz_patterns = {
            "bar": r"막대.*차트|bar.*chart|막대.*그래프",
            "line": r"선.*차트|line.*chart|선.*그래프|시계열",
            "scatter": r"산점도|scatter|점.*그래프",
            "pie": r"원.*차트|pie.*chart|파이.*차트",
            "histogram": r"히스토그램|histogram|분포.*차트",
            "box": r"박스.*플롯|box.*plot|상자.*그림",
            "heatmap": r"히트맵|heatmap|열.*지도"
        }
        
        for viz_type, pattern in viz_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return viz_type
        
        return "auto"  # 자동 선택
    
    def generate_analysis_plan(self, intent: QueryIntent, dataframe_info: Dict) -> List[Dict[str, Any]]:
        """분석 계획 생성"""
        plan = []
        
        # 쿼리 유형별 분석 단계 정의
        if intent.query_type == QueryType.SUMMARY:
            plan.extend([
                {"step": "basic_info", "description": "데이터 기본 정보 확인"},
                {"step": "data_types", "description": "데이터 타입 분석"},
                {"step": "sample_data", "description": "샘플 데이터 표시"}
            ])
        
        elif intent.query_type == QueryType.STATISTICS:
            plan.extend([
                {"step": "descriptive_stats", "description": "기술통계 계산"},
                {"step": "distribution_analysis", "description": "분포 분석"}
            ])
            if intent.target_columns:
                plan.append({
                    "step": "column_stats", 
                    "description": f"지정 컬럼 통계: {', '.join(intent.target_columns)}"
                })
        
        elif intent.query_type == QueryType.CORRELATION:
            plan.extend([
                {"step": "correlation_matrix", "description": "상관관계 매트릭스 계산"},
                {"step": "high_correlations", "description": "높은 상관관계 식별"}
            ])
        
        elif intent.query_type == QueryType.MISSING_DATA:
            plan.extend([
                {"step": "missing_data_count", "description": "결측 데이터 개수 확인"},
                {"step": "missing_data_pattern", "description": "결측 데이터 패턴 분석"}
            ])
        
        elif intent.query_type == QueryType.VISUALIZATION:
            plan.extend([
                {"step": "data_preparation", "description": "시각화용 데이터 준비"},
                {"step": "chart_generation", "description": f"{intent.visualization_type} 차트 생성"}
            ])
        
        elif intent.query_type == QueryType.GROUPBY:
            if intent.target_columns:
                plan.extend([
                    {"step": "group_analysis", "description": f"그룹별 분석: {', '.join(intent.target_columns)}"},
                    {"step": "group_statistics", "description": "그룹별 통계 계산"}
                ])
        
        else:
            # 기본 분석 계획
            plan.extend([
                {"step": "general_analysis", "description": "일반적인 데이터 분석"},
                {"step": "basic_statistics", "description": "기본 통계 정보"}
            ])
        
        return plan
    
    def format_analysis_result(self, intent: QueryIntent, results: Dict[str, Any]) -> str:
        """분석 결과 포맷팅"""
        try:
            # 쿼리 유형별 결과 포맷팅
            if intent.query_type == QueryType.SUMMARY:
                return self._format_summary_result(results)
            elif intent.query_type == QueryType.STATISTICS:
                return self._format_statistics_result(results)
            elif intent.query_type == QueryType.CORRELATION:
                return self._format_correlation_result(results)
            elif intent.query_type == QueryType.MISSING_DATA:
                return self._format_missing_data_result(results)
            else:
                return self._format_general_result(results)
                
        except Exception as e:
            logger.error(f"❌ 결과 포맷팅 실패: {e}")
            return f"분석이 완료되었지만 결과 포맷팅 중 오류가 발생했습니다: {str(e)}"
    
    def _format_summary_result(self, results: Dict[str, Any]) -> str:
        """요약 결과 포맷팅"""
        return f"""# 📊 **데이터 요약**

{results.get('summary', '요약 정보를 생성할 수 없습니다.')}

## 🔍 **주요 특징**
{results.get('key_features', '특징 정보가 없습니다.')}
"""
    
    def _format_statistics_result(self, results: Dict[str, Any]) -> str:
        """통계 결과 포맷팅"""
        return f"""# 📈 **통계 분석 결과**

{results.get('statistics', '통계 정보를 생성할 수 없습니다.')}

## 🎯 **주요 인사이트**
{results.get('insights', '인사이트를 생성할 수 없습니다.')}
"""
    
    def _format_correlation_result(self, results: Dict[str, Any]) -> str:
        """상관관계 결과 포맷팅"""
        return f"""# 🔗 **상관관계 분석**

{results.get('correlation', '상관관계 정보를 생성할 수 없습니다.')}

## ⭐ **높은 상관관계**
{results.get('high_correlations', '높은 상관관계가 발견되지 않았습니다.')}
"""
    
    def _format_missing_data_result(self, results: Dict[str, Any]) -> str:
        """결측 데이터 결과 포맷팅"""
        return f"""# 🔍 **결측 데이터 분석**

{results.get('missing_analysis', '결측 데이터 분석을 수행할 수 없습니다.')}

## 💡 **권장사항**
{results.get('recommendations', '권장사항이 없습니다.')}
"""
    
    def _format_general_result(self, results: Dict[str, Any]) -> str:
        """일반 결과 포맷팅"""
        return f"""# 📊 **분석 결과**

{results.get('analysis', '분석 결과를 표시할 수 없습니다.')}

## 📋 **상세 정보**
{results.get('details', '상세 정보가 없습니다.')}
""" 