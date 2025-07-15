"""
LLM First 데이터 엔진 (LLM First Data Engine)

pandas_agent의 LLMEngine 패턴을 기준으로 한 지능형 데이터 처리 엔진
모든 데이터 관련 결정을 LLM이 동적으로 수행하는 핵심 엔진

핵심 원칙:
- LLM First: 하드코딩 없이 LLM이 모든 결정 수행
- 동적 적응: 다양한 데이터 소스와 형식에 대응
- 지능형 최적화: 컨텍스트 기반 최적 전략 수립
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd

# A2A 및 LLM 관련 import
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .unified_data_interface import DataIntent, DataIntentType, LoadingStrategy, A2AContext
from ..utils.file_scanner import FileScanner
from ..utils.encoding_detector import EncodingDetector

logger = logging.getLogger(__name__)


class LLMFirstDataEngine:
    """
    LLM First 데이터 처리 엔진
    
    pandas_agent의 LLMEngine 패턴을 확장하여
    데이터 로딩의 모든 결정을 LLM이 지능적으로 수행
    """
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        LLM First 데이터 엔진 초기화
        
        Args:
            model_name: 사용할 LLM 모델명 (기본: 환경변수 기반)
            api_key: API 키 (기본: 환경변수 기반)
        """
        self.model_name = model_name or self._get_default_model()
        self.api_key = api_key or self._get_api_key()
        self.llm = self._initialize_llm()
        
        # 보조 도구 초기화
        self.file_scanner = FileScanner()
        self.encoding_detector = EncodingDetector()
        
        # 내부 캐시
        self._intent_cache: Dict[str, DataIntent] = {}
        self._file_analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"✅ LLMFirstDataEngine 초기화 완료 (모델: {self.model_name})")
    
    def _get_default_model(self) -> str:
        """기본 LLM 모델 결정 (환경변수 기반)"""
        if os.getenv('OPENAI_API_KEY'):
            return "gpt-4o-mini"
        elif os.getenv('ANTHROPIC_API_KEY'):
            return "claude-3-haiku-20240307"
        elif os.getenv('GOOGLE_API_KEY'):
            return "gemini-pro"
        else:
            # OLLAMA 폴백 (프로젝트 메모리 기준)
            return "ollama/gemma2:4b"
    
    def _get_api_key(self) -> Optional[str]:
        """API 키 추출"""
        if "openai" in self.model_name.lower() or "gpt" in self.model_name.lower():
            return os.getenv('OPENAI_API_KEY')
        elif "claude" in self.model_name.lower() or "anthropic" in self.model_name.lower():
            return os.getenv('ANTHROPIC_API_KEY')
        elif "gemini" in self.model_name.lower():
            return os.getenv('GOOGLE_API_KEY')
        return None
    
    def _initialize_llm(self):
        """LLM 초기화"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("⚠️ LangChain 없음. Mock LLM 사용")
            return None
        
        try:
            if "openai" in self.model_name.lower() or "gpt" in self.model_name.lower():
                return ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=0.3
                )
            elif "claude" in self.model_name.lower() or "anthropic" in self.model_name.lower():
                return ChatAnthropic(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=0.3
                )
            else:
                # OLLAMA 또는 기타 모델 (기본 OpenAI 인터페이스 사용)
                return ChatOpenAI(
                    model=self.model_name,
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    temperature=0.3
                )
        except Exception as e:
            logger.error(f"❌ LLM 초기화 실패: {e}")
            return None
    
    async def analyze_intent(self, user_query: str, context: A2AContext) -> DataIntent:
        """
        사용자 의도 분석 (LLM First)
        
        Args:
            user_query: 사용자 쿼리
            context: A2A 컨텍스트
            
        Returns:
            DataIntent: LLM이 분석한 데이터 처리 의도
        """
        # 캐시 확인
        cache_key = f"{hash(user_query)}_{context.session_id}"
        if cache_key in self._intent_cache:
            logger.info("✅ 의도 분석 캐시 히트")
            return self._intent_cache[cache_key]
        
        try:
            if not self.llm:
                # 폴백: 간단한 키워드 기반 분석
                return self._fallback_intent_analysis(user_query)
            
            # LLM 프롬프트 구성
            system_prompt = """You are an intelligent data analysis intent analyzer for CherryAI.
            Analyze the user's request and determine their data processing intent.
            
            Available intent types:
            - analysis: General data analysis and insights
            - visualization: Data visualization and charts
            - cleaning: Data cleaning and preprocessing
            - transformation: Data transformation and manipulation
            - modeling: Machine learning and statistical modeling
            - feature_engineering: Feature creation and selection
            - sql_query: SQL database operations
            - reporting: Report generation and summarization
            - eda: Exploratory Data Analysis
            - orchestration: Multi-agent coordination
            
            Respond in JSON format with:
            {
                "intent_type": "intent_name",
                "confidence": 0.0-1.0,
                "file_preferences": ["specific file patterns or names"],
                "operations": ["list of required operations"],
                "constraints": {"key": "value"},
                "priority": 1-5,
                "requires_visualization": true/false,
                "estimated_complexity": "low/medium/high"
            }"""
            
            user_prompt = f"""
            User Query: {user_query}
            
            Context Information:
            - Session ID: {context.session_id}
            - Request ID: {context.request_id}
            - Timestamp: {context.to_dict().get('timestamp')}
            
            Analyze this query and provide the intent analysis in JSON format.
            """
            
            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # JSON 파싱
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            intent_data = json.loads(response_text)
            
            # DataIntent 객체 생성
            intent = DataIntent(
                intent_type=DataIntentType(intent_data['intent_type']),
                confidence=intent_data['confidence'],
                file_preferences=intent_data['file_preferences'],
                operations=intent_data['operations'],
                constraints=intent_data['constraints'],
                priority=intent_data.get('priority', 1),
                requires_visualization=intent_data.get('requires_visualization', False),
                estimated_complexity=intent_data.get('estimated_complexity', 'medium')
            )
            
            # 캐시 저장
            self._intent_cache[cache_key] = intent
            
            logger.info(f"✅ 의도 분석 완료: {intent.intent_type.value} (신뢰도: {intent.confidence:.2f})")
            return intent
            
        except Exception as e:
            logger.error(f"❌ LLM 의도 분석 실패: {e}")
            return self._fallback_intent_analysis(user_query)
    
    def _fallback_intent_analysis(self, user_query: str) -> DataIntent:
        """
        폴백 의도 분석 (다국어 키워드 기반)
        
        pandas_agent 패턴을 기준으로 한 지능형 키워드 매칭
        우선순위: 특수 작업 > 일반 분석
        """
        query_lower = user_query.lower()
        
        # 다국어 키워드 매핑 (우선순위 순서)
        intent_keywords = {
            DataIntentType.VISUALIZATION: [
                # 영어
                'plot', 'chart', 'graph', 'visualize', 'visualization', 'show', 'display',
                'scatter', 'histogram', 'bar', 'pie', 'line', 'heatmap', 'boxplot',
                # 한국어
                '시각화', '그래프', '차트', '플롯', '보여', '표시', '그림'
            ],
            DataIntentType.CLEANING: [
                # 영어  
                'clean', 'cleaning', 'missing', 'null', 'duplicate', 'remove',
                'fill', 'drop', 'handle', 'preprocess', 'preprocessing',
                # 한국어
                '정리', '청소', '클리닝', '결측', '중복', '제거', '처리'
            ],
            DataIntentType.MODELING: [
                # 영어
                'model', 'modeling', 'predict', 'prediction', 'machine learning', 'ml',
                'train', 'training', 'algorithm', 'classifier', 'regression',
                # 한국어
                '모델', '모델링', '예측', '머신러닝', '학습', '훈련', '알고리즘'
            ],
            DataIntentType.FEATURE_ENGINEERING: [
                # 영어
                'feature', 'features', 'engineer', 'engineering', 'create', 'generate',
                'transform', 'encode', 'scale', 'normalize',
                # 한국어
                '피처', '특성', '특징', '엔지니어링', '생성', '변환', '인코딩'
            ],
            DataIntentType.SQL_QUERY: [
                # 영어
                'sql', 'query', 'database', 'select', 'join', 'where', 'group by',
                'table', 'column', 'row',
                # 한국어
                '쿼리', '데이터베이스', '테이블', '컬럼', '조회', '검색'
            ],
            DataIntentType.REPORTING: [
                # 영어
                'report', 'summary', 'summarize', 'document', 'generate',
                'export', 'output', 'write',
                # 한국어
                '보고서', '리포트', '요약', '정리', '문서', '생성', '출력'
            ],
            DataIntentType.EDA: [
                # 영어 - 구체적인 EDA 키워드만
                'explore', 'exploration', 'eda', 'exploratory', 'statistics',
                'describe', 'correlation', 'distribution',
                # 한국어 - 구체적인 EDA 키워드만  
                '탐색', '탐색적', '통계', '기술통계', '상관관계', '분포'
            ],
            DataIntentType.TRANSFORMATION: [
                # 영어
                'transform', 'transformation', 'convert', 'change', 'modify',
                'reshape', 'pivot', 'melt', 'groupby',
                # 한국어
                '변환', '전환', '바꾸', '수정', '재구성', '피벗'
            ]
        }
        
        # 키워드 매칭 점수 계산 (우선순위 고려)
        intent_scores = {}
        
        for intent_type, keywords in intent_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in query_lower:
                    # 길이가 긴 키워드일수록 높은 점수
                    keyword_score = len(keyword) / 10.0
                    score += keyword_score
                    matched_keywords.append(keyword)
            
            if score > 0:
                intent_scores[intent_type] = {
                    'score': score,
                    'keywords': matched_keywords
                }
        
        # 가장 높은 점수의 의도 선택
        if intent_scores:
            best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k]['score'])
            confidence = min(0.9, 0.5 + intent_scores[best_intent]['score'] * 0.1)  # 0.5-0.9 범위
            
            # 작업 목록 생성
            operations = self._generate_operations_for_intent(best_intent, intent_scores[best_intent]['keywords'])
            
            return DataIntent(
                intent_type=best_intent,
                confidence=confidence,
                file_preferences=[],
                operations=operations,
                constraints={},
                priority=self._get_intent_priority(best_intent),
                requires_visualization=best_intent == DataIntentType.VISUALIZATION or 'visual' in query_lower,
                estimated_complexity=self._estimate_complexity(query_lower, best_intent)
            )
        
        # 키워드 매칭 실패 시 기본 분석
        return DataIntent(
            intent_type=DataIntentType.ANALYSIS,
            confidence=0.7,
            file_preferences=[],
            operations=['analyze'],
            constraints={},
            priority=2,
            requires_visualization='visual' in query_lower or '시각' in query_lower,
            estimated_complexity='medium'
        )
    
    def _generate_operations_for_intent(self, intent_type: DataIntentType, matched_keywords: List[str]) -> List[str]:
        """의도에 따른 작업 목록 생성"""
        base_operations = {
            DataIntentType.VISUALIZATION: ['load', 'visualize', 'plot'],
            DataIntentType.CLEANING: ['load', 'clean', 'validate'],
            DataIntentType.MODELING: ['load', 'preprocess', 'train', 'evaluate'],
            DataIntentType.FEATURE_ENGINEERING: ['load', 'engineer', 'transform'],
            DataIntentType.SQL_QUERY: ['connect', 'query', 'fetch'],
            DataIntentType.REPORTING: ['load', 'analyze', 'generate_report'],
            DataIntentType.EDA: ['load', 'explore', 'describe'],
            DataIntentType.TRANSFORMATION: ['load', 'transform', 'reshape'],
            DataIntentType.ANALYSIS: ['load', 'analyze']
        }
        
        operations = base_operations.get(intent_type, ['analyze'])
        
        # 키워드 기반 세부 작업 추가
        if any(kw in matched_keywords for kw in ['correlation', '상관관계']):
            operations.append('correlation_analysis')
        if any(kw in matched_keywords for kw in ['distribution', '분포']):
            operations.append('distribution_analysis')
        
        return operations
    
    def _get_intent_priority(self, intent_type: DataIntentType) -> int:
        """의도별 우선순위 반환"""
        priority_map = {
            DataIntentType.ORCHESTRATION: 5,  # 최고 우선순위
            DataIntentType.REPORTING: 4,
            DataIntentType.MODELING: 3,
            DataIntentType.FEATURE_ENGINEERING: 3,
            DataIntentType.VISUALIZATION: 2,
            DataIntentType.EDA: 2,
            DataIntentType.CLEANING: 2,
            DataIntentType.TRANSFORMATION: 2,
            DataIntentType.SQL_QUERY: 2,
            DataIntentType.ANALYSIS: 1  # 기본 우선순위
        }
        return priority_map.get(intent_type, 1)
    
    def _estimate_complexity(self, query: str, intent_type: DataIntentType) -> str:
        """작업 복잡도 추정"""
        complex_keywords = ['multiple', 'complex', 'advanced', '복잡', '고급', '다중']
        medium_keywords = ['analyze', 'process', '분석', '처리']
        
        if any(kw in query.lower() for kw in complex_keywords):
            return 'high'
        elif any(kw in query.lower() for kw in medium_keywords):
            return 'medium'
        elif intent_type in [DataIntentType.MODELING, DataIntentType.FEATURE_ENGINEERING]:
            return 'high'
        elif intent_type in [DataIntentType.VISUALIZATION, DataIntentType.CLEANING]:
            return 'medium'
        else:
            return 'low'
    
    async def select_optimal_file(self, intent: DataIntent, available_files: List[str]) -> str:
        """
        최적 파일 선택 (LLM First)
        
        Args:
            intent: 데이터 처리 의도
            available_files: 사용 가능한 파일 리스트
            
        Returns:
            str: 선택된 파일 경로
        """
        if not available_files:
            raise ValueError("사용 가능한 파일이 없습니다")
        
        if len(available_files) == 1:
            return available_files[0]
        
        try:
            if not self.llm:
                # 폴백: 파일 확장자와 크기 기반 선택
                return self._fallback_file_selection(intent, available_files)
            
            # 파일 정보 수집
            file_infos = []
            for file_path in available_files:
                info = await self._analyze_file_info(file_path)
                file_infos.append(info)
            
            # LLM 프롬프트 구성
            system_prompt = """You are an intelligent file selector for data analysis.
            Select the most appropriate file based on the user's intent and file characteristics.
            
            Consider:
            - File size and complexity
            - Data format compatibility
            - Intent requirements
            - File naming patterns
            - Potential data quality
            
            Respond with just the file path of the selected file."""
            
            user_prompt = f"""
            User Intent: {intent.intent_type.value}
            Operations: {intent.operations}
            File Preferences: {intent.file_preferences}
            Estimated Complexity: {intent.estimated_complexity}
            
            Available Files:
            {json.dumps(file_infos, indent=2, ensure_ascii=False)}
            
            Select the most appropriate file for this analysis task.
            Return only the file path.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            selected_file = response.content.strip()
            
            # 선택된 파일이 유효한지 확인
            if selected_file in available_files:
                logger.info(f"✅ LLM 파일 선택: {selected_file}")
                return selected_file
            else:
                # LLM 응답이 부정확한 경우 폴백
                return self._fallback_file_selection(intent, available_files)
                
        except Exception as e:
            logger.error(f"❌ LLM 파일 선택 실패: {e}")
            return self._fallback_file_selection(intent, available_files)
    
    def _fallback_file_selection(self, intent: DataIntent, available_files: List[str]) -> str:
        """폴백 파일 선택"""
        # 파일 확장자 우선순위
        priority_extensions = {
            DataIntentType.ANALYSIS: ['.csv', '.xlsx', '.json'],
            DataIntentType.VISUALIZATION: ['.csv', '.xlsx'],
            DataIntentType.MODELING: ['.csv', '.parquet'],
            DataIntentType.EDA: ['.csv', '.xlsx']
        }
        
        preferred_exts = priority_extensions.get(intent.intent_type, ['.csv', '.xlsx', '.json'])
        
        # 우선순위에 따라 파일 선택
        for ext in preferred_exts:
            for file_path in available_files:
                if file_path.lower().endswith(ext):
                    return file_path
        
        # 기본값: 첫 번째 파일
        return available_files[0]
    
    async def _analyze_file_info(self, file_path: str) -> Dict[str, Any]:
        """파일 정보 분석"""
        cache_key = file_path
        if cache_key in self._file_analysis_cache:
            return self._file_analysis_cache[cache_key]
        
        try:
            file_obj = Path(file_path)
            info = {
                "path": file_path,
                "name": file_obj.name,
                "extension": file_obj.suffix,
                "size_mb": file_obj.stat().st_size / (1024 * 1024) if file_obj.exists() else 0,
                "exists": file_obj.exists()
            }
            
            # 인코딩 감지 (텍스트 파일인 경우)
            if info["extension"] in ['.csv', '.txt']:
                encoding = await self.encoding_detector.detect_encoding(file_path)
                info["encoding"] = encoding
            
            # 캐시 저장
            self._file_analysis_cache[cache_key] = info
            return info
            
        except Exception as e:
            logger.error(f"❌ 파일 정보 분석 실패 {file_path}: {e}")
            return {
                "path": file_path,
                "name": Path(file_path).name,
                "extension": Path(file_path).suffix,
                "size_mb": 0,
                "exists": False,
                "error": str(e)
            }
    
    async def create_loading_strategy(self, file_path: str, intent: DataIntent) -> LoadingStrategy:
        """
        데이터 로딩 전략 생성 (LLM First)
        
        Args:
            file_path: 파일 경로
            intent: 데이터 처리 의도
            
        Returns:
            LoadingStrategy: 최적화된 로딩 전략
        """
        try:
            file_info = await self._analyze_file_info(file_path)
            
            if not self.llm:
                # 폴백: 휴리스틱 기반 전략
                return self._create_fallback_strategy(file_info, intent)
            
            # LLM 프롬프트 구성
            system_prompt = """You are an intelligent data loading strategy optimizer.
            Create an optimal loading strategy based on file characteristics and analysis intent.
            
            Consider:
            - File size and memory constraints
            - Data analysis requirements
            - Performance optimization
            - Error handling needs
            
            Respond in JSON format:
            {
                "encoding": "recommended_encoding",
                "chunk_size": null_or_number,
                "sample_ratio": null_or_float,
                "use_cache": true/false,
                "cache_ttl": seconds,
                "fallback_encodings": ["list", "of", "encodings"],
                "preprocessing_steps": ["list", "of", "steps"]
            }"""
            
            user_prompt = f"""
            File Information:
            {json.dumps(file_info, indent=2, ensure_ascii=False)}
            
            Analysis Intent:
            - Type: {intent.intent_type.value}
            - Complexity: {intent.estimated_complexity}
            - Operations: {intent.operations}
            - Requires Visualization: {intent.requires_visualization}
            
            Create an optimal loading strategy for this scenario.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()
            
            # JSON 파싱
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            strategy_data = json.loads(response_text)
            
            strategy = LoadingStrategy(
                encoding=strategy_data.get('encoding', 'utf-8'),
                chunk_size=strategy_data.get('chunk_size'),
                sample_ratio=strategy_data.get('sample_ratio'),
                use_cache=strategy_data.get('use_cache', True),
                cache_ttl=strategy_data.get('cache_ttl', 3600),
                fallback_encodings=strategy_data.get('fallback_encodings'),
                preprocessing_steps=strategy_data.get('preprocessing_steps')
            )
            
            logger.info(f"✅ LLM 로딩 전략 생성: {strategy.encoding}, 캐시={strategy.use_cache}")
            return strategy
            
        except Exception as e:
            logger.error(f"❌ LLM 로딩 전략 생성 실패: {e}")
            return self._create_fallback_strategy(file_info, intent)
    
    def _create_fallback_strategy(self, file_info: Dict[str, Any], intent: DataIntent) -> LoadingStrategy:
        """폴백 로딩 전략"""
        size_mb = file_info.get('size_mb', 0)
        
        # 파일 크기에 따른 전략
        if size_mb > 500:  # 500MB 이상
            return LoadingStrategy(
                encoding='utf-8',
                chunk_size=10000,
                sample_ratio=0.1,
                use_cache=True,
                cache_ttl=7200
            )
        elif size_mb > 100:  # 100MB 이상
            return LoadingStrategy(
                encoding='utf-8',
                chunk_size=5000,
                sample_ratio=None,
                use_cache=True,
                cache_ttl=3600
            )
        else:  # 100MB 미만
            return LoadingStrategy(
                encoding='utf-8',
                chunk_size=None,
                sample_ratio=None,
                use_cache=True,
                cache_ttl=1800
            )
    
    async def scan_available_files(self, context: A2AContext) -> List[str]:
        """사용 가능한 파일 스캔"""
        return await self.file_scanner.scan_data_files(context.session_id)
    
    def clear_cache(self):
        """캐시 정리"""
        self._intent_cache.clear()
        self._file_analysis_cache.clear()
        logger.info("✅ LLMFirstDataEngine 캐시 정리 완료") 