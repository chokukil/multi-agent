# core/execution/timeout_manager.py
from pydantic import BaseModel, Field
from typing import Dict, Optional
from enum import Enum
import re

class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    INTENSIVE = "intensive"

class TimeoutConfig(BaseModel):
    """타임아웃 설정 모델"""
    simple_timeout: int = Field(default=30, ge=10, le=300)
    moderate_timeout: int = Field(default=120, ge=30, le=600)
    complex_timeout: int = Field(default=300, ge=60, le=1200)
    intensive_timeout: int = Field(default=600, ge=120, le=3600)
    
    # 에이전트별 가중치
    agent_multipliers: Dict[str, float] = Field(default_factory=lambda: {
        "eda_specialist": 1.5,
        "visualization_expert": 2.0,
        "statistical_analyst": 1.8,
        "data_preprocessor": 1.3,
        "final_responder": 1.2,
        "planner": 1.1,
        "executor": 1.4,
    })

class TimeoutManager:
    """지능형 타임아웃 관리자"""
    
    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TimeoutConfig()
        
        # 복잡도 판별 키워드
        self.complexity_keywords = {
            TaskComplexity.SIMPLE: [
                "안녕", "고마워", "감사", "hello", "hi", "thanks", "thank you",
                "status", "상태", "현재", "어떻게", "정보"
            ],
            TaskComplexity.MODERATE: [
                "분석", "analysis", "show", "보여줘", "확인", "check", 
                "summary", "요약", "기본", "basic"
            ],
            TaskComplexity.COMPLEX: [
                "시각화", "visualization", "차트", "그래프", "plot", "차이",
                "비교", "compare", "correlation", "상관관계", "패턴", "pattern"
            ],
            TaskComplexity.INTENSIVE: [
                "머신러닝", "machine learning", "모델", "model", "예측", "predict",
                "최적화", "optimization", "클러스터", "cluster", "딥러닝", "deep learning",
                "알고리즘", "algorithm", "복잡한", "complex", "상세한", "detailed"
            ]
        }
    
    def analyze_query_complexity(self, query: str) -> Dict:
        """
        쿼리를 분석하여 복잡도를 판별
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            복잡도 정보가 담긴 딕셔너리
        """
        query_lower = query.lower()
        scores = {complexity: 0 for complexity in TaskComplexity}
        
        # 키워드 기반 점수 계산
        for complexity, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[complexity] += 1
        
        # 쿼리 길이 기반 보정
        query_length = len(query)
        if query_length > 100:
            scores[TaskComplexity.COMPLEX] += 1
        if query_length > 200:
            scores[TaskComplexity.INTENSIVE] += 1
            
        # 복잡한 구문 패턴 검사
        complex_patterns = [
            r'동시에|함께|병렬로',  # 여러 작업 동시 수행
            r'단계별로|순서대로|차례로',  # 다단계 작업
            r'비교.*분석|분석.*비교',  # 비교 분석
            r'최적.*방법|최선.*방법',  # 최적화 요청
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, query_lower):
                scores[TaskComplexity.COMPLEX] += 1
        
        # 최고 점수를 받은 복잡도 선택
        best_complexity = max(scores, key=scores.get)
        
        # 점수가 0인 경우 기본값 설정
        if scores[best_complexity] == 0:
            best_complexity = TaskComplexity.MODERATE
            
        return {
            'complexity': best_complexity,
            'scores': scores,
            'confidence': min(scores[best_complexity] / max(1, len(query.split())), 1.0),
            'reasoning': f"Keywords matched: {scores[best_complexity]}, Length: {query_length}"
        }
    
    def calculate_timeout(self, complexity: TaskComplexity, 
                         agent_type: str = None) -> int:
        """
        복잡도와 에이전트 타입에 따른 타임아웃 계산
        
        Args:
            complexity: 작업 복잡도
            agent_type: 에이전트 타입
            
        Returns:
            계산된 타임아웃 (초)
        """
        return self.get_timeout(complexity, agent_type)
    
    def get_timeout(self, complexity: TaskComplexity, 
                   agent_type: str = None) -> int:
        """작업 복잡도와 에이전트 타입에 따른 타임아웃 계산"""
        base_timeout = {
            TaskComplexity.SIMPLE: self.config.simple_timeout,
            TaskComplexity.MODERATE: self.config.moderate_timeout,
            TaskComplexity.COMPLEX: self.config.complex_timeout,
            TaskComplexity.INTENSIVE: self.config.intensive_timeout,
        }[complexity]
        
        # 에이전트별 가중치 적용
        if agent_type:
            # 에이전트 타입을 소문자로 변환하여 매칭
            agent_key = agent_type.lower().replace('_', '')
            for key, multiplier in self.config.agent_multipliers.items():
                if key.replace('_', '') in agent_key or agent_key in key.replace('_', ''):
                    return int(base_timeout * multiplier)
        
        return base_timeout
    
    def get_timeout_by_query_type(self, query_complexity: str) -> int:
        """쿼리 복잡도에 따른 타임아웃 반환"""
        complexity_mapping = {
            "simple": TaskComplexity.SIMPLE,
            "moderate": TaskComplexity.MODERATE, 
            "complex": TaskComplexity.COMPLEX,
            "intensive": TaskComplexity.INTENSIVE
        }
        
        complexity = complexity_mapping.get(query_complexity, TaskComplexity.COMPLEX)
        return self.get_timeout(complexity)