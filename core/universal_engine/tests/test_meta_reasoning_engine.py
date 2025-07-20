"""
Meta-Reasoning Engine 단위 테스트

요구사항 2.1, 2.2, 2.3에 따른 메타 추론 엔진 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from ..meta_reasoning_engine import MetaReasoningEngine


class TestMetaReasoningEngine:
    """MetaReasoningEngine 테스트 클래스"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """LLM 클라이언트 모킹"""
        mock_client = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def meta_engine(self, mock_llm_client):
        """테스트용 MetaReasoningEngine 인스턴스"""
        with patch('core.universal_engine.meta_reasoning_engine.LLMFactory.create_llm', return_value=mock_llm_client):
            engine = MetaReasoningEngine()
            return engine
    
    @pytest.mark.asyncio
    async def test_initial_observation_stage(self, meta_engine, mock_llm_client):
        """
        초기 관찰 단계 테스트
        """
        # 준비
        test_query = "이 데이터에서 이상한 패턴이 있나요?"
        test_data = {"columns": ["value1", "value2"], "rows": 100}
        
        expected_response = """```json
        {
            "data_observations": "100행의 데이터에 value1, value2 컬럼이 있음",
            "query_intent": "이상 패턴 감지 요청",
            "potential_blind_spots": "데이터 품질, 시간적 맥락",
            "domain_context": "데이터 분석",
            "data_characteristics": {
                "type": "structured",
                "structure": "tabular",
                "patterns": "unknown"
            }
        }
        ```"""
        
        mock_llm_client.agenerate.return_value = expected_response
        
        # 실행
        result = await meta_engine._perform_initial_observation(test_query, test_data)
        
        # 검증
        assert result["data_observations"] == "100행의 데이터에 value1, value2 컬럼이 있음"
        assert result["query_intent"] == "이상 패턴 감지 요청"
        assert "데이터 품질" in result["potential_blind_spots"]
        assert mock_llm_client.agenerate.called
    
    @pytest.mark.asyncio
    async def test_multi_perspective_analysis_stage(self, meta_engine, mock_llm_client):
        """
        다각도 분석 단계 테스트
        """
        # 준비
        initial_analysis = {
            "query_intent": "이상 패턴 감지 요청",
            "domain_context": "데이터 분석"
        }
        
        expected_response = """```json
        {
            "alternative_approaches": ["통계적 분석", "시각화", "머신러닝"],
            "expert_perspective": {
                "expectations": "정확한 통계적 근거",
                "technical_depth": "high",
                "key_insights": ["분포 분석", "이상치 탐지"]
            },
            "beginner_perspective": {
                "needs": "직관적 설명",
                "simplifications": "복잡한 통계 용어 피하기",
                "guidance": "단계별 해석"
            },
            "estimated_user_level": "intermediate",
            "recommended_approach": "시각화 + 기본 통계"
        }
        ```"""
        
        mock_llm_client.agenerate.return_value = expected_response
        
        # 실행
        result = await meta_engine._perform_multi_perspective_analysis(
            initial_analysis, "test query", "test data"
        )
        
        # 검증
        assert len(result["alternative_approaches"]) == 3
        assert result["estimated_user_level"] == "intermediate"
        assert result["expert_perspective"]["technical_depth"] == "high"
        assert result["beginner_perspective"]["needs"] == "직관적 설명"
    
    @pytest.mark.asyncio
    async def test_self_verification_stage(self, meta_engine, mock_llm_client):
        """
        자가 검증 단계 테스트
        """
        # 준비
        multi_perspective = {
            "estimated_user_level": "intermediate",
            "recommended_approach": "시각화 + 기본 통계"
        }
        
        expected_response = """```json
        {
            "logical_consistency": {
                "is_consistent": true,
                "inconsistencies": []
            },
            "practical_value": {
                "is_helpful": true,
                "value_points": ["명확한 방향 제시", "사용자 수준 고려"],
                "limitations": ["구체적 방법론 부족"]
            },
            "uncertainties": {
                "high_confidence_areas": ["사용자 수준 추정"],
                "low_confidence_areas": ["정확한 이상 패턴 유형"],
                "clarification_needed": ["이상 패턴의 정의"]
            },
            "overall_confidence": 0.7
        }
        ```"""
        
        mock_llm_client.agenerate.return_value = expected_response
        
        # 실행
        result = await meta_engine._perform_self_verification(multi_perspective)
        
        # 검증
        assert result["logical_consistency"]["is_consistent"] is True
        assert result["practical_value"]["is_helpful"] is True
        assert result["overall_confidence"] == 0.7
        assert "사용자 수준 추정" in result["uncertainties"]["high_confidence_areas"]
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_determination(self, meta_engine, mock_llm_client):
        """
        적응적 응답 전략 결정 단계 테스트
        """
        # 준비
        self_verification = {
            "overall_confidence": 0.7,
            "uncertainties": {
                "clarification_needed": ["이상 패턴의 정의"]
            }
        }
        context = {"user_history": []}
        
        expected_response = """```json
        {
            "response_strategy": {
                "approach": "점진적",
                "explanation_depth": "medium",
                "technical_level": "medium",
                "interaction_style": "educational"
            },
            "content_structure": {
                "present_confidently": ["데이터 구조 분석"],
                "seek_clarification": ["이상 패턴의 정의"],
                "progressive_disclosure": ["기본 분석", "상세 분석", "고급 해석"]
            },
            "estimated_user_profile": {
                "expertise_level": "intermediate",
                "learning_style": "visual",
                "domain_familiarity": "medium"
            },
            "follow_up_recommendations": ["시각화 생성", "통계 분석", "패턴 검증"]
        }
        ```"""
        
        mock_llm_client.agenerate.return_value = expected_response
        
        # 실행
        result = await meta_engine._determine_adaptive_strategy(self_verification, context)
        
        # 검증
        assert result["response_strategy"]["approach"] == "점진적"
        assert result["estimated_user_profile"]["expertise_level"] == "intermediate"
        assert len(result["follow_up_recommendations"]) == 3
    
    @pytest.mark.asyncio
    async def test_meta_rewarding_quality_assessment(self, meta_engine, mock_llm_client):
        """
        메타 보상 패턴을 통한 분석 품질 평가 테스트
        """
        # 준비
        response_strategy = {
            "response_strategy": {"approach": "점진적"},
            "estimated_user_profile": {"expertise_level": "intermediate"}
        }
        
        expected_response = """```json
        {
            "accuracy_score": 0.8,
            "completeness_score": 0.7,
            "appropriateness_score": 0.9,
            "clarity_score": 0.8,
            "practicality_score": 0.75,
            "overall_quality": 0.8,
            "confidence": 0.8,
            "improvements_needed": ["더 구체적인 방법론 제시"],
            "strengths": ["사용자 수준 고려", "단계적 접근"],
            "next_steps": ["명확화 질문", "데이터 탐색"]
        }
        ```"""
        
        mock_llm_client.agenerate.return_value = expected_response
        
        # 실행
        result = await meta_engine._assess_analysis_quality(response_strategy)
        
        # 검증
        assert result["overall_quality"] == 0.8
        assert result["accuracy_score"] == 0.8
        assert "사용자 수준 고려" in result["strengths"]
        assert "더 구체적인 방법론 제시" in result["improvements_needed"]
    
    @pytest.mark.asyncio
    async def test_complete_meta_reasoning_pipeline(self, meta_engine, mock_llm_client):
        """
        완전한 메타 추론 파이프라인 테스트
        """
        # 준비
        test_query = "우리 공장의 생산 데이터에서 품질 문제를 찾을 수 있나요?"
        test_data = {"production_data": "sample"}
        test_context = {"session_id": "test123"}
        
        # 각 단계별 응답 설정
        responses = [
            # Initial observation
            """```json
            {"data_observations": "생산 데이터 분석", "query_intent": "품질 문제 감지", "domain_context": "제조업"}
            ```""",
            # Multi-perspective analysis  
            """```json
            {"estimated_user_level": "expert", "recommended_approach": "통계적 품질 관리"}
            ```""",
            # Self verification
            """```json
            {"overall_confidence": 0.85, "logical_consistency": {"is_consistent": true}}
            ```""",
            # Adaptive strategy
            """```json
            {"response_strategy": {"approach": "직접적"}, "estimated_user_profile": {"expertise_level": "expert"}}
            ```""",
            # Quality assessment
            """```json
            {"overall_quality": 0.9, "confidence": 0.85}
            ```"""
        ]
        
        mock_llm_client.agenerate.side_effect = responses
        
        # 실행
        result = await meta_engine.analyze_request(test_query, test_data, test_context)
        
        # 검증
        assert "initial_analysis" in result
        assert "multi_perspective" in result
        assert "self_verification" in result
        assert "response_strategy" in result
        assert "quality_assessment" in result
        assert result["confidence_level"] == 0.85
        assert result["user_profile"]["expertise_level"] == "expert"
        
        # LLM 호출 횟수 확인 (5번의 단계)
        assert mock_llm_client.agenerate.call_count == 5
    
    def test_data_characteristics_analysis(self, meta_engine):
        """
        데이터 특성 분석 테스트
        """
        # 테스트 데이터들
        test_cases = [
            # DataFrame-like 데이터
            ({"shape": (100, 5), "columns": ["a", "b", "c", "d", "e"]}, "DataFrame"),
            # 리스트 데이터
            ([1, 2, 3, 4, 5], "list"),
            # 딕셔너리 데이터
            ({"key1": "value1", "key2": "value2"}, "dict")
        ]
        
        for test_data, expected_type in test_cases:
            result = meta_engine._analyze_data_characteristics(test_data)
            characteristics = json.loads(result)
            assert expected_type.lower() in characteristics["type"].lower()
    
    def test_json_response_parsing(self, meta_engine):
        """
        JSON 응답 파싱 테스트
        """
        # 정상적인 JSON 블록
        valid_response = """```json
        {
            "test_key": "test_value",
            "number": 42
        }
        ```"""
        
        result = meta_engine._parse_json_response(valid_response)
        assert result["test_key"] == "test_value"
        assert result["number"] == 42
        
        # 잘못된 JSON
        invalid_response = "This is not JSON"
        result = meta_engine._parse_json_response(invalid_response)
        assert "raw_response" in result
        assert "parse_error" in result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, meta_engine, mock_llm_client):
        """
        오류 처리 테스트
        """
        # LLM 클라이언트 오류 시뮬레이션
        mock_llm_client.agenerate.side_effect = Exception("LLM connection error")
        
        # 오류가 제대로 전파되는지 확인
        with pytest.raises(Exception) as exc_info:
            await meta_engine.analyze_request("test query", "test data", {})
        
        assert "LLM connection error" in str(exc_info.value)
    
    def test_reasoning_patterns_initialization(self, meta_engine):
        """
        추론 패턴 초기화 테스트
        """
        patterns = meta_engine.reasoning_patterns
        
        # 필수 패턴들이 모두 로드되었는지 확인
        required_patterns = ['self_reflection', 'meta_rewarding', 'chain_of_thought', 'zero_shot_adaptive']
        for pattern in required_patterns:
            assert pattern in patterns
            assert isinstance(patterns[pattern], str)
            assert len(patterns[pattern]) > 0
    
    def test_confidence_calculation_edge_cases(self, meta_engine):
        """
        신뢰도 계산 경계 사례 테스트
        """
        # 빈 데이터
        result = meta_engine._analyze_data_characteristics(None)
        assert "unknown" in result.lower() or "none" in result.lower()
        
        # 크기가 0인 데이터
        result = meta_engine._analyze_data_characteristics([])
        characteristics = json.loads(result)
        assert "0" in characteristics["size"] or "empty" in characteristics["size"]