"""
메타 러닝 시스템 (Meta Learning System)
Phase 3.3: 분석 경험 학습 및 지속적 개선

핵심 기능:
- 분석 패턴 학습 및 축적
- 성공/실패 사례 분류 및 학습
- 자동 전략 개선 및 최적화
- 사용자 피드백 통합 학습
- 적응형 분석 파라미터 조정
- 도메인별 지식 축적
"""

import asyncio
import json
import logging
import pickle
import statistics
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AnalysisOutcome(Enum):
    """분석 결과 평가"""
    EXCELLENT = "excellent"     # 매우 성공적
    GOOD = "good"              # 성공적
    ACCEPTABLE = "acceptable"   # 보통
    POOR = "poor"              # 미흡
    FAILED = "failed"          # 실패

class LearningType(Enum):
    """학습 유형"""
    PATTERN_RECOGNITION = "pattern_recognition"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    PARAMETER_TUNING = "parameter_tuning"
    FEEDBACK_INTEGRATION = "feedback_integration"
    DOMAIN_ADAPTATION = "domain_adaptation"

@dataclass
class AnalysisExperience:
    """분석 경험 데이터"""
    id: str
    timestamp: datetime
    data_profile: Dict[str, Any]
    strategy_used: Dict[str, Any]
    outcome: AnalysisOutcome
    user_feedback: Dict[str, Any]
    performance_metrics: Dict[str, float]
    insights_quality: float
    execution_time: float
    context: str
    domain_tags: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

@dataclass
class LearningRule:
    """학습 규칙"""
    id: str
    rule_type: LearningType
    condition: Dict[str, Any]
    action: Dict[str, Any]
    confidence: float
    success_count: int
    failure_count: int
    last_updated: datetime
    domain_applicability: List[str] = field(default_factory=list)

@dataclass
class MetaKnowledge:
    """메타 지식"""
    data_patterns: Dict[str, Any]
    successful_strategies: Dict[str, List[Dict[str, Any]]]
    common_pitfalls: List[Dict[str, Any]]
    optimal_parameters: Dict[str, Dict[str, Any]]
    domain_insights: Dict[str, Dict[str, Any]]
    adaptation_rules: List[LearningRule]

class ExperienceDatabase:
    """경험 데이터베이스"""
    
    def __init__(self, db_path: str = "core/meta_learning/experience.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 경험 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    data_profile TEXT,
                    strategy_used TEXT,
                    outcome TEXT,
                    user_feedback TEXT,
                    performance_metrics TEXT,
                    insights_quality REAL,
                    execution_time REAL,
                    context TEXT,
                    domain_tags TEXT,
                    lessons_learned TEXT
                )
            """)
            
            # 학습 규칙 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_rules (
                    id TEXT PRIMARY KEY,
                    rule_type TEXT,
                    condition TEXT,
                    action TEXT,
                    confidence REAL,
                    success_count INTEGER,
                    failure_count INTEGER,
                    last_updated DATETIME,
                    domain_applicability TEXT
                )
            """)
            
            # 성능 메트릭 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_tracking (
                    timestamp DATETIME,
                    metric_name TEXT,
                    metric_value REAL,
                    context TEXT,
                    PRIMARY KEY (timestamp, metric_name, context)
                )
            """)
            
            conn.commit()
    
    def store_experience(self, experience: AnalysisExperience):
        """경험 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO experiences 
                (id, timestamp, data_profile, strategy_used, outcome, user_feedback,
                 performance_metrics, insights_quality, execution_time, context,
                 domain_tags, lessons_learned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.id,
                experience.timestamp,
                json.dumps(experience.data_profile),
                json.dumps(experience.strategy_used),
                experience.outcome.value,
                json.dumps(experience.user_feedback),
                json.dumps(experience.performance_metrics),
                experience.insights_quality,
                experience.execution_time,
                experience.context,
                json.dumps(experience.domain_tags),
                json.dumps(experience.lessons_learned)
            ))
            
            conn.commit()
    
    def get_experiences(self, 
                       outcome_filter: Optional[List[AnalysisOutcome]] = None,
                       domain_filter: Optional[List[str]] = None,
                       limit: int = 1000) -> List[AnalysisExperience]:
        """경험 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM experiences"
            params = []
            conditions = []
            
            if outcome_filter:
                outcome_values = [outcome.value for outcome in outcome_filter]
                conditions.append(f"outcome IN ({','.join(['?'] * len(outcome_values))})")
                params.extend(outcome_values)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            experiences = []
            for row in rows:
                experience = AnalysisExperience(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    data_profile=json.loads(row[2]),
                    strategy_used=json.loads(row[3]),
                    outcome=AnalysisOutcome(row[4]),
                    user_feedback=json.loads(row[5]),
                    performance_metrics=json.loads(row[6]),
                    insights_quality=row[7],
                    execution_time=row[8],
                    context=row[9],
                    domain_tags=json.loads(row[10]),
                    lessons_learned=json.loads(row[11])
                )
                experiences.append(experience)
            
            return experiences
    
    def store_learning_rule(self, rule: LearningRule):
        """학습 규칙 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO learning_rules 
                (id, rule_type, condition, action, confidence, success_count,
                 failure_count, last_updated, domain_applicability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.id,
                rule.rule_type.value,
                json.dumps(rule.condition),
                json.dumps(rule.action),
                rule.confidence,
                rule.success_count,
                rule.failure_count,
                rule.last_updated,
                json.dumps(rule.domain_applicability)
            ))
            
            conn.commit()
    
    def get_learning_rules(self, rule_type: Optional[LearningType] = None) -> List[LearningRule]:
        """학습 규칙 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if rule_type:
                cursor.execute("SELECT * FROM learning_rules WHERE rule_type = ?", (rule_type.value,))
            else:
                cursor.execute("SELECT * FROM learning_rules")
            
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule = LearningRule(
                    id=row[0],
                    rule_type=LearningType(row[1]),
                    condition=json.loads(row[2]),
                    action=json.loads(row[3]),
                    confidence=row[4],
                    success_count=row[5],
                    failure_count=row[6],
                    last_updated=datetime.fromisoformat(row[7]),
                    domain_applicability=json.loads(row[8])
                )
                rules.append(rule)
            
            return rules

class PatternLearner:
    """패턴 학습기"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clusterer = KMeans(n_clusters=10, random_state=42)
        self.pattern_cache = {}
        
    def learn_data_patterns(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """데이터 패턴 학습"""
        logger.info(f"📊 데이터 패턴 학습 시작: {len(experiences)}개 경험")
        
        patterns = {
            "size_patterns": self._learn_size_patterns(experiences),
            "type_patterns": self._learn_type_patterns(experiences),
            "quality_patterns": self._learn_quality_patterns(experiences),
            "complexity_patterns": self._learn_complexity_patterns(experiences)
        }
        
        return patterns
    
    def _learn_size_patterns(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """데이터 크기 패턴 학습"""
        size_outcomes = defaultdict(list)
        
        for exp in experiences:
            data_profile = exp.data_profile
            if 'shape' in data_profile:
                rows, cols = data_profile['shape']
                size_category = self._categorize_size(rows, cols)
                size_outcomes[size_category].append(exp.outcome.value)
        
        # 크기별 성공 패턴 분석
        size_success_rates = {}
        for size_cat, outcomes in size_outcomes.items():
            success_count = sum(1 for outcome in outcomes if outcome in ['excellent', 'good'])
            success_rate = success_count / len(outcomes) if outcomes else 0
            size_success_rates[size_cat] = success_rate
        
        return {
            "size_success_rates": size_success_rates,
            "optimal_sizes": [cat for cat, rate in size_success_rates.items() if rate > 0.7],
            "challenging_sizes": [cat for cat, rate in size_success_rates.items() if rate < 0.3]
        }
    
    def _categorize_size(self, rows: int, cols: int) -> str:
        """데이터 크기 분류"""
        if rows < 1000 and cols < 10:
            return "small"
        elif rows < 10000 and cols < 50:
            return "medium"
        elif rows < 100000 and cols < 100:
            return "large"
        else:
            return "very_large"
    
    def _learn_type_patterns(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """데이터 타입 패턴 학습"""
        type_outcomes = defaultdict(list)
        
        for exp in experiences:
            data_profile = exp.data_profile
            if 'data_characteristics' in data_profile:
                characteristics = data_profile['data_characteristics']
                char_signature = "_".join(sorted(characteristics))
                type_outcomes[char_signature].append(exp.outcome.value)
        
        # 타입별 성공 패턴
        type_success_rates = {}
        for char_sig, outcomes in type_outcomes.items():
            success_count = sum(1 for outcome in outcomes if outcome in ['excellent', 'good'])
            success_rate = success_count / len(outcomes) if outcomes else 0
            type_success_rates[char_sig] = success_rate
        
        return {
            "type_success_rates": type_success_rates,
            "preferred_types": [t for t, rate in type_success_rates.items() if rate > 0.6],
            "difficult_types": [t for t, rate in type_success_rates.items() if rate < 0.4]
        }
    
    def _learn_quality_patterns(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """데이터 품질 패턴 학습"""
        quality_bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]
        quality_outcomes = {f"{low}-{high}": [] for low, high in quality_bins}
        
        for exp in experiences:
            data_profile = exp.data_profile
            quality_score = data_profile.get('quality_score', 0.5)
            
            for low, high in quality_bins:
                if low <= quality_score < high:
                    quality_outcomes[f"{low}-{high}"].append(exp.outcome.value)
                    break
        
        # 품질별 성공 패턴
        quality_success_rates = {}
        for quality_range, outcomes in quality_outcomes.items():
            if outcomes:
                success_count = sum(1 for outcome in outcomes if outcome in ['excellent', 'good'])
                success_rate = success_count / len(outcomes)
                quality_success_rates[quality_range] = success_rate
        
        return {
            "quality_success_rates": quality_success_rates,
            "quality_threshold": 0.6  # 경험적 임계값
        }
    
    def _learn_complexity_patterns(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """복잡도 패턴 학습"""
        complexity_outcomes = defaultdict(list)
        
        for exp in experiences:
            data_profile = exp.data_profile
            complexity = data_profile.get('complexity_level', 'medium')
            complexity_outcomes[complexity].append(exp.outcome.value)
        
        complexity_success_rates = {}
        for complexity, outcomes in complexity_outcomes.items():
            success_count = sum(1 for outcome in outcomes if outcome in ['excellent', 'good'])
            success_rate = success_count / len(outcomes) if outcomes else 0
            complexity_success_rates[complexity] = success_rate
        
        return {
            "complexity_success_rates": complexity_success_rates,
            "manageable_complexity": [c for c, rate in complexity_success_rates.items() if rate > 0.5]
        }

class StrategyOptimizer:
    """전략 최적화기"""
    
    def __init__(self):
        self.strategy_performance = defaultdict(list)
        self.parameter_optimization = {}
        
    def optimize_strategies(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """전략 최적화"""
        logger.info(f"🎯 전략 최적화 시작: {len(experiences)}개 경험 분석")
        
        optimization_results = {
            "strategy_rankings": self._rank_strategies(experiences),
            "context_strategies": self._optimize_context_strategies(experiences),
            "parameter_recommendations": self._optimize_parameters(experiences),
            "hybrid_strategies": self._discover_hybrid_strategies(experiences)
        }
        
        return optimization_results
    
    def _rank_strategies(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """전략 순위 매기기"""
        strategy_performance = defaultdict(list)
        
        for exp in experiences:
            strategy = exp.strategy_used.get('context', 'unknown')
            
            # 성능 점수 계산 (결과 + 품질 + 실행시간)
            outcome_score = self._outcome_to_score(exp.outcome)
            quality_score = exp.insights_quality
            time_penalty = max(0, 1 - (exp.execution_time / 300))  # 5분 기준
            
            overall_score = (outcome_score * 0.5 + quality_score * 0.3 + time_penalty * 0.2)
            strategy_performance[strategy].append(overall_score)
        
        # 전략별 평균 성능
        strategy_rankings = {}
        for strategy, scores in strategy_performance.items():
            avg_score = statistics.mean(scores)
            confidence = 1 / (1 + statistics.stdev(scores)) if len(scores) > 1 else 0.5
            strategy_rankings[strategy] = {
                "avg_score": avg_score,
                "confidence": confidence,
                "sample_size": len(scores)
            }
        
        # 순위별 정렬
        sorted_strategies = sorted(strategy_rankings.items(), 
                                 key=lambda x: x[1]['avg_score'], reverse=True)
        
        return {
            "rankings": dict(sorted_strategies),
            "top_strategies": [s[0] for s in sorted_strategies[:3]],
            "strategies_to_avoid": [s[0] for s in sorted_strategies[-2:] if s[1]['avg_score'] < 0.3]
        }
    
    def _outcome_to_score(self, outcome: AnalysisOutcome) -> float:
        """결과를 점수로 변환"""
        score_mapping = {
            AnalysisOutcome.EXCELLENT: 1.0,
            AnalysisOutcome.GOOD: 0.8,
            AnalysisOutcome.ACCEPTABLE: 0.6,
            AnalysisOutcome.POOR: 0.3,
            AnalysisOutcome.FAILED: 0.0
        }
        return score_mapping.get(outcome, 0.5)
    
    def _optimize_context_strategies(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """컨텍스트별 전략 최적화"""
        context_strategies = defaultdict(lambda: defaultdict(list))
        
        # 컨텍스트별 전략 성능 분석
        for exp in experiences:
            context = exp.context
            strategy = exp.strategy_used.get('context', 'unknown')
            score = self._outcome_to_score(exp.outcome)
            
            context_strategies[context][strategy].append(score)
        
        # 컨텍스트별 최적 전략 선정
        optimal_strategies = {}
        for context, strategies in context_strategies.items():
            strategy_scores = {}
            for strategy, scores in strategies.items():
                strategy_scores[strategy] = statistics.mean(scores)
            
            if strategy_scores:
                best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
                optimal_strategies[context] = {
                    "strategy": best_strategy[0],
                    "score": best_strategy[1],
                    "alternatives": sorted(strategy_scores.items(), 
                                        key=lambda x: x[1], reverse=True)[1:3]
                }
        
        return optimal_strategies
    
    def _optimize_parameters(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """파라미터 최적화"""
        parameter_performance = defaultdict(lambda: defaultdict(list))
        
        for exp in experiences:
            strategy = exp.strategy_used
            score = self._outcome_to_score(exp.outcome)
            
            # 전략의 파라미터 분석
            if 'adaptive_parameters' in strategy:
                params = strategy['adaptive_parameters']
                for param_name, param_value in params.items():
                    parameter_performance[param_name][str(param_value)].append(score)
        
        # 파라미터별 최적값 찾기
        optimal_parameters = {}
        for param_name, value_scores in parameter_performance.items():
            if value_scores:
                avg_scores = {value: statistics.mean(scores) 
                            for value, scores in value_scores.items()}
                best_value = max(avg_scores.items(), key=lambda x: x[1])
                optimal_parameters[param_name] = {
                    "optimal_value": best_value[0],
                    "score": best_value[1],
                    "alternatives": sorted(avg_scores.items(), 
                                        key=lambda x: x[1], reverse=True)[1:3]
                }
        
        return optimal_parameters
    
    def _discover_hybrid_strategies(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """하이브리드 전략 발견"""
        # 성공적인 경험들의 공통 패턴 찾기
        successful_experiences = [exp for exp in experiences 
                                if exp.outcome in [AnalysisOutcome.EXCELLENT, AnalysisOutcome.GOOD]]
        
        if len(successful_experiences) < 5:
            return {"hybrid_strategies": [], "confidence": 0.0}
        
        # 성공적인 전략의 조합 패턴 분석
        strategy_combinations = defaultdict(int)
        
        for exp in successful_experiences:
            strategy = exp.strategy_used
            techniques = strategy.get('techniques', [])
            steps = strategy.get('priority_steps', [])
            
            # 기법과 단계의 조합 패턴
            combo_key = "_".join(sorted(techniques[:3])) + "|" + "_".join(sorted(steps[:3]))
            strategy_combinations[combo_key] += 1
        
        # 빈도수 기반 하이브리드 전략 제안
        common_combinations = sorted(strategy_combinations.items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
        
        hybrid_strategies = []
        for combo, frequency in common_combinations:
            techniques_part, steps_part = combo.split("|")
            hybrid_strategies.append({
                "techniques": techniques_part.split("_"),
                "steps": steps_part.split("_"),
                "frequency": frequency,
                "confidence": frequency / len(successful_experiences)
            })
        
        return {
            "hybrid_strategies": hybrid_strategies,
            "discovery_confidence": 0.7 if hybrid_strategies else 0.0
        }

class FeedbackIntegrator:
    """피드백 통합기"""
    
    def __init__(self):
        self.feedback_weights = {
            "user_satisfaction": 0.4,
            "insight_relevance": 0.3,
            "actionability": 0.2,
            "novelty": 0.1
        }
        
    def integrate_feedback(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """피드백 통합 학습"""
        logger.info(f"📝 피드백 통합 학습: {len(experiences)}개 경험")
        
        feedback_analysis = {
            "satisfaction_patterns": self._analyze_satisfaction_patterns(experiences),
            "improvement_suggestions": self._extract_improvement_suggestions(experiences),
            "quality_correlations": self._analyze_quality_correlations(experiences),
            "user_preferences": self._learn_user_preferences(experiences)
        }
        
        return feedback_analysis
    
    def _analyze_satisfaction_patterns(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """만족도 패턴 분석"""
        satisfaction_data = []
        
        for exp in experiences:
            feedback = exp.user_feedback
            if 'satisfaction_score' in feedback:
                satisfaction_data.append({
                    'satisfaction': feedback['satisfaction_score'],
                    'strategy': exp.strategy_used.get('context', 'unknown'),
                    'quality': exp.insights_quality,
                    'time': exp.execution_time
                })
        
        if not satisfaction_data:
            return {"patterns": [], "insights": ["피드백 데이터 부족"]}
        
        # 만족도 영향 요인 분석
        high_satisfaction = [d for d in satisfaction_data if d['satisfaction'] > 0.8]
        low_satisfaction = [d for d in satisfaction_data if d['satisfaction'] < 0.4]
        
        patterns = []
        
        if high_satisfaction:
            common_strategies = set(d['strategy'] for d in high_satisfaction)
            avg_quality = statistics.mean(d['quality'] for d in high_satisfaction)
            avg_time = statistics.mean(d['time'] for d in high_satisfaction)
            
            patterns.append({
                "type": "high_satisfaction",
                "strategies": list(common_strategies),
                "avg_quality": avg_quality,
                "avg_time": avg_time,
                "count": len(high_satisfaction)
            })
        
        if low_satisfaction:
            problem_strategies = set(d['strategy'] for d in low_satisfaction)
            patterns.append({
                "type": "low_satisfaction",
                "strategies": list(problem_strategies),
                "count": len(low_satisfaction)
            })
        
        return {
            "patterns": patterns,
            "insights": self._generate_satisfaction_insights(patterns)
        }
    
    def _generate_satisfaction_insights(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """만족도 인사이트 생성"""
        insights = []
        
        for pattern in patterns:
            if pattern["type"] == "high_satisfaction":
                insights.append(f"고만족 전략: {', '.join(pattern['strategies'][:3])}")
                insights.append(f"고만족 품질 평균: {pattern['avg_quality']:.2f}")
            elif pattern["type"] == "low_satisfaction":
                insights.append(f"개선 필요 전략: {', '.join(pattern['strategies'][:3])}")
        
        return insights
    
    def _extract_improvement_suggestions(self, experiences: List[AnalysisExperience]) -> List[str]:
        """개선 제안 추출"""
        suggestions = []
        
        for exp in experiences:
            feedback = exp.user_feedback
            if 'suggestions' in feedback and feedback['suggestions']:
                suggestions.extend(feedback['suggestions'])
        
        # 빈도수 기반 중요 제안 추출
        suggestion_counts = defaultdict(int)
        for suggestion in suggestions:
            suggestion_counts[suggestion] += 1
        
        top_suggestions = sorted(suggestion_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        return [suggestion for suggestion, count in top_suggestions]
    
    def _analyze_quality_correlations(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """품질 상관관계 분석"""
        correlations = {}
        
        quality_scores = []
        satisfaction_scores = []
        execution_times = []
        
        for exp in experiences:
            if 'satisfaction_score' in exp.user_feedback:
                quality_scores.append(exp.insights_quality)
                satisfaction_scores.append(exp.user_feedback['satisfaction_score'])
                execution_times.append(exp.execution_time)
        
        if len(quality_scores) > 3:
            # 간단한 상관관계 계산
            quality_satisfaction_corr = np.corrcoef(quality_scores, satisfaction_scores)[0, 1]
            quality_time_corr = np.corrcoef(quality_scores, execution_times)[0, 1]
            
            correlations = {
                "quality_satisfaction": quality_satisfaction_corr,
                "quality_time": quality_time_corr,
                "insights": [
                    f"품질-만족도 상관관계: {quality_satisfaction_corr:.3f}",
                    f"품질-실행시간 상관관계: {quality_time_corr:.3f}"
                ]
            }
        
        return correlations
    
    def _learn_user_preferences(self, experiences: List[AnalysisExperience]) -> Dict[str, Any]:
        """사용자 선호도 학습"""
        preferences = {
            "preferred_analysis_depth": "medium",
            "preferred_visualization_types": [],
            "preferred_insight_format": "detailed",
            "time_tolerance": 180  # 3분 기본값
        }
        
        # 피드백에서 선호도 패턴 추출
        depth_preferences = []
        time_tolerances = []
        
        for exp in experiences:
            feedback = exp.user_feedback
            
            if 'preferred_depth' in feedback:
                depth_preferences.append(feedback['preferred_depth'])
                
            if 'time_acceptable' in feedback and feedback['time_acceptable']:
                time_tolerances.append(exp.execution_time)
        
        if depth_preferences:
            preferences["preferred_analysis_depth"] = max(set(depth_preferences), 
                                                        key=depth_preferences.count)
        
        if time_tolerances:
            preferences["time_tolerance"] = statistics.median(time_tolerances)
        
        return preferences

class MetaLearningSystem:
    """메타 러닝 시스템 (통합)"""
    
    def __init__(self):
        self.db = ExperienceDatabase()
        self.pattern_learner = PatternLearner()
        self.strategy_optimizer = StrategyOptimizer()
        self.feedback_integrator = FeedbackIntegrator()
        
        # 메타 지식 저장소
        self.meta_knowledge = MetaKnowledge(
            data_patterns={},
            successful_strategies={},
            common_pitfalls=[],
            optimal_parameters={},
            domain_insights={},
            adaptation_rules=[]
        )
        
        # 학습 설정
        self.learning_interval = timedelta(hours=24)  # 24시간마다 학습
        self.min_experiences_for_learning = 10
        self.confidence_threshold = 0.7
        
        # 결과 저장 경로
        self.knowledge_path = Path("core/meta_learning/meta_knowledge.pkl")
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 기존 지식 로드
        self._load_meta_knowledge()
    
    async def record_analysis_experience(self, 
                                       data_profile: Dict[str, Any],
                                       strategy_used: Dict[str, Any],
                                       outcome: AnalysisOutcome,
                                       user_feedback: Dict[str, Any],
                                       performance_metrics: Dict[str, float],
                                       insights_quality: float,
                                       execution_time: float,
                                       context: str = "") -> str:
        """분석 경험 기록"""
        
        experience_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data_profile))}"
        
        experience = AnalysisExperience(
            id=experience_id,
            timestamp=datetime.now(),
            data_profile=data_profile,
            strategy_used=strategy_used,
            outcome=outcome,
            user_feedback=user_feedback,
            performance_metrics=performance_metrics,
            insights_quality=insights_quality,
            execution_time=execution_time,
            context=context,
            domain_tags=self._extract_domain_tags(data_profile, context),
            lessons_learned=self._extract_lessons_learned(outcome, user_feedback)
        )
        
        # 데이터베이스에 저장
        self.db.store_experience(experience)
        
        logger.info(f"📝 분석 경험 기록: {experience_id} (결과: {outcome.value})")
        
        # 즉시 학습 트리거 (비동기)
        asyncio.create_task(self._trigger_learning_if_needed())
        
        return experience_id
    
    async def _trigger_learning_if_needed(self):
        """필요시 학습 트리거"""
        recent_experiences = self.db.get_experiences(limit=self.min_experiences_for_learning + 1)
        
        if len(recent_experiences) >= self.min_experiences_for_learning:
            await self.learn_from_experiences()
    
    async def learn_from_experiences(self) -> Dict[str, Any]:
        """경험으로부터 학습"""
        logger.info("🧠 메타 러닝 시작...")
        
        # 최근 경험 수집
        experiences = self.db.get_experiences(limit=1000)
        
        if len(experiences) < self.min_experiences_for_learning:
            logger.warning(f"학습을 위한 경험 부족: {len(experiences)}/{self.min_experiences_for_learning}")
            return {"status": "insufficient_data"}
        
        learning_results = {}
        
        # 1. 패턴 학습
        try:
            data_patterns = self.pattern_learner.learn_data_patterns(experiences)
            self.meta_knowledge.data_patterns.update(data_patterns)
            learning_results["pattern_learning"] = data_patterns
            logger.info("✅ 패턴 학습 완료")
        except Exception as e:
            logger.error(f"패턴 학습 실패: {e}")
            learning_results["pattern_learning"] = {"error": str(e)}
        
        # 2. 전략 최적화
        try:
            strategy_optimization = self.strategy_optimizer.optimize_strategies(experiences)
            self.meta_knowledge.successful_strategies.update(strategy_optimization)
            learning_results["strategy_optimization"] = strategy_optimization
            logger.info("✅ 전략 최적화 완료")
        except Exception as e:
            logger.error(f"전략 최적화 실패: {e}")
            learning_results["strategy_optimization"] = {"error": str(e)}
        
        # 3. 피드백 통합
        try:
            feedback_analysis = self.feedback_integrator.integrate_feedback(experiences)
            learning_results["feedback_integration"] = feedback_analysis
            logger.info("✅ 피드백 통합 완료")
        except Exception as e:
            logger.error(f"피드백 통합 실패: {e}")
            learning_results["feedback_integration"] = {"error": str(e)}
        
        # 4. 학습 규칙 생성
        new_rules = await self._generate_learning_rules(experiences, learning_results)
        for rule in new_rules:
            self.db.store_learning_rule(rule)
        learning_results["new_rules"] = len(new_rules)
        
        # 5. 메타 지식 저장
        self._save_meta_knowledge()
        
        learning_results["learning_timestamp"] = datetime.now().isoformat()
        learning_results["experiences_processed"] = len(experiences)
        
        logger.info(f"🎯 메타 러닝 완료: {len(experiences)}개 경험 처리, {len(new_rules)}개 새 규칙 생성")
        
        return learning_results
    
    async def _generate_learning_rules(self, experiences: List[AnalysisExperience], 
                                     learning_results: Dict[str, Any]) -> List[LearningRule]:
        """학습 규칙 생성"""
        rules = []
        
        # 패턴 기반 규칙
        if "pattern_learning" in learning_results:
            patterns = learning_results["pattern_learning"]
            
            # 크기 기반 규칙
            if "size_patterns" in patterns:
                size_patterns = patterns["size_patterns"]
                for size_cat in size_patterns.get("optimal_sizes", []):
                    rule = LearningRule(
                        id=f"size_rule_{size_cat}_{datetime.now().strftime('%Y%m%d')}",
                        rule_type=LearningType.PATTERN_RECOGNITION,
                        condition={"size_category": size_cat},
                        action={"recommend": "apply_standard_strategy"},
                        confidence=0.8,
                        success_count=0,
                        failure_count=0,
                        last_updated=datetime.now()
                    )
                    rules.append(rule)
        
        # 전략 기반 규칙
        if "strategy_optimization" in learning_results:
            strategy_opt = learning_results["strategy_optimization"]
            
            if "top_strategies" in strategy_opt:
                for strategy in strategy_opt["top_strategies"][:2]:  # 상위 2개
                    rule = LearningRule(
                        id=f"strategy_rule_{strategy}_{datetime.now().strftime('%Y%m%d')}",
                        rule_type=LearningType.STRATEGY_OPTIMIZATION,
                        condition={"general_analysis": True},
                        action={"prefer_strategy": strategy},
                        confidence=0.7,
                        success_count=0,
                        failure_count=0,
                        last_updated=datetime.now()
                    )
                    rules.append(rule)
        
        return rules
    
    def get_recommendations(self, data_profile: Dict[str, Any], 
                          context: str = "") -> Dict[str, Any]:
        """메타 지식 기반 추천"""
        recommendations = {
            "strategy_recommendations": [],
            "parameter_suggestions": {},
            "warnings": [],
            "confidence": 0.5
        }
        
        # 데이터 패턴 기반 추천
        if self.meta_knowledge.data_patterns:
            pattern_recs = self._get_pattern_recommendations(data_profile)
            recommendations["strategy_recommendations"].extend(pattern_recs)
        
        # 성공 전략 기반 추천
        if self.meta_knowledge.successful_strategies:
            strategy_recs = self._get_strategy_recommendations(data_profile, context)
            recommendations["strategy_recommendations"].extend(strategy_recs)
        
        # 학습 규칙 적용
        rule_recs = self._apply_learning_rules(data_profile, context)
        recommendations.update(rule_recs)
        
        # 추천 신뢰도 계산
        recommendations["confidence"] = self._calculate_recommendation_confidence(
            data_profile, recommendations
        )
        
        return recommendations
    
    def _get_pattern_recommendations(self, data_profile: Dict[str, Any]) -> List[str]:
        """패턴 기반 추천"""
        recommendations = []
        
        patterns = self.meta_knowledge.data_patterns
        
        # 크기 패턴 체크
        if 'shape' in data_profile and 'size_patterns' in patterns:
            rows, cols = data_profile['shape']
            size_category = self.pattern_learner._categorize_size(rows, cols)
            
            size_patterns = patterns['size_patterns']
            if size_category in size_patterns.get('challenging_sizes', []):
                recommendations.append(f"주의: {size_category} 크기 데이터는 분석이 어려울 수 있습니다")
            elif size_category in size_patterns.get('optimal_sizes', []):
                recommendations.append(f"적절한 크기: {size_category} 데이터로 좋은 결과가 예상됩니다")
        
        return recommendations
    
    def _get_strategy_recommendations(self, data_profile: Dict[str, Any], context: str) -> List[str]:
        """전략 기반 추천"""
        recommendations = []
        
        strategies = self.meta_knowledge.successful_strategies
        
        if 'strategy_rankings' in strategies:
            rankings = strategies['strategy_rankings']
            if 'top_strategies' in rankings:
                top_strategy = rankings['top_strategies'][0] if rankings['top_strategies'] else None
                if top_strategy:
                    recommendations.append(f"추천 전략: {top_strategy} (높은 성공률)")
        
        return recommendations
    
    def _apply_learning_rules(self, data_profile: Dict[str, Any], context: str) -> Dict[str, Any]:
        """학습 규칙 적용"""
        applicable_rules = self.db.get_learning_rules()
        
        applied_recommendations = {
            "rule_based_suggestions": [],
            "parameter_adjustments": {}
        }
        
        for rule in applicable_rules:
            if self._rule_applies(rule, data_profile, context):
                if rule.confidence > self.confidence_threshold:
                    if rule.rule_type == LearningType.STRATEGY_OPTIMIZATION:
                        action = rule.action
                        if 'prefer_strategy' in action:
                            applied_recommendations["rule_based_suggestions"].append(
                                f"규칙 기반 추천: {action['prefer_strategy']} 전략 사용"
                            )
                    elif rule.rule_type == LearningType.PARAMETER_TUNING:
                        applied_recommendations["parameter_adjustments"].update(rule.action)
        
        return applied_recommendations
    
    def _rule_applies(self, rule: LearningRule, data_profile: Dict[str, Any], context: str) -> bool:
        """규칙 적용 가능성 확인"""
        condition = rule.condition
        
        # 간단한 조건 매칭
        if 'size_category' in condition:
            if 'shape' in data_profile:
                rows, cols = data_profile['shape']
                size_category = self.pattern_learner._categorize_size(rows, cols)
                return size_category == condition['size_category']
        
        if 'general_analysis' in condition:
            return condition['general_analysis']
        
        return False
    
    def _calculate_recommendation_confidence(self, data_profile: Dict[str, Any], 
                                          recommendations: Dict[str, Any]) -> float:
        """추천 신뢰도 계산"""
        confidence_factors = []
        
        # 메타 지식 풍부도
        knowledge_richness = len(self.meta_knowledge.data_patterns) * 0.1
        confidence_factors.append(min(1.0, knowledge_richness))
        
        # 추천 수량
        rec_count = len(recommendations.get("strategy_recommendations", []))
        rec_factor = min(1.0, rec_count * 0.2)
        confidence_factors.append(rec_factor)
        
        # 데이터 프로파일 완성도
        profile_completeness = len(data_profile) * 0.1
        confidence_factors.append(min(1.0, profile_completeness))
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    def _extract_domain_tags(self, data_profile: Dict[str, Any], context: str) -> List[str]:
        """도메인 태그 추출"""
        tags = []
        
        # 컨텍스트 기반 태그
        if context:
            tags.append(context.lower())
        
        # 데이터 특성 기반 태그
        if 'data_characteristics' in data_profile:
            tags.extend(data_profile['data_characteristics'])
        
        # 크기 기반 태그
        if 'shape' in data_profile:
            rows, cols = data_profile['shape']
            size_tag = self.pattern_learner._categorize_size(rows, cols)
            tags.append(size_tag)
        
        return list(set(tags))  # 중복 제거
    
    def _extract_lessons_learned(self, outcome: AnalysisOutcome, 
                               user_feedback: Dict[str, Any]) -> List[str]:
        """교훈 추출"""
        lessons = []
        
        if outcome == AnalysisOutcome.FAILED:
            lessons.append("분석 실패 사례 - 전략 재검토 필요")
        elif outcome == AnalysisOutcome.EXCELLENT:
            lessons.append("성공적인 분석 사례 - 전략 재사용 권장")
        
        if 'suggestions' in user_feedback:
            lessons.extend(user_feedback['suggestions'])
        
        return lessons
    
    def _load_meta_knowledge(self):
        """메타 지식 로드"""
        if self.knowledge_path.exists():
            try:
                with open(self.knowledge_path, 'rb') as f:
                    self.meta_knowledge = pickle.load(f)
                logger.info("📚 기존 메타 지식 로드 완료")
            except Exception as e:
                logger.warning(f"메타 지식 로드 실패: {e}")
    
    def _save_meta_knowledge(self):
        """메타 지식 저장"""
        try:
            with open(self.knowledge_path, 'wb') as f:
                pickle.dump(self.meta_knowledge, f)
            logger.info("💾 메타 지식 저장 완료")
        except Exception as e:
            logger.error(f"메타 지식 저장 실패: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """학습 상태 반환"""
        experiences = self.db.get_experiences(limit=100)
        recent_experiences = [exp for exp in experiences 
                            if (datetime.now() - exp.timestamp).days < 7]
        
        return {
            "total_experiences": len(experiences),
            "recent_experiences": len(recent_experiences),
            "knowledge_areas": len(self.meta_knowledge.data_patterns),
            "learning_rules": len(self.db.get_learning_rules()),
            "last_learning": datetime.now().isoformat(),  # 실제로는 마지막 학습 시점
            "confidence_level": 0.7  # 현재 시스템 신뢰도
        }


# 사용 예시 및 테스트
async def test_meta_learning_system():
    """메타 러닝 시스템 테스트"""
    meta_system = MetaLearningSystem()
    
    # 테스트 경험 기록
    test_data_profile = {
        "shape": (1000, 20),
        "data_characteristics": ["numerical", "mixed"],
        "quality_score": 0.8,
        "complexity_level": "medium"
    }
    
    test_strategy = {
        "context": "exploration",
        "techniques": ["descriptive_statistics", "correlation_analysis"],
        "priority_steps": ["data_overview", "pattern_discovery"]
    }
    
    test_feedback = {
        "satisfaction_score": 0.9,
        "suggestions": ["더 상세한 시각화 필요"]
    }
    
    # 경험 기록
    experience_id = await meta_system.record_analysis_experience(
        data_profile=test_data_profile,
        strategy_used=test_strategy,
        outcome=AnalysisOutcome.GOOD,
        user_feedback=test_feedback,
        performance_metrics={"accuracy": 0.85},
        insights_quality=0.8,
        execution_time=120.0,
        context="business_analytics"
    )
    
    print(f"📝 경험 기록됨: {experience_id}")
    
    # 추천 받기
    recommendations = meta_system.get_recommendations(test_data_profile, "business_analytics")
    
    print(f"\n💡 메타 러닝 추천:")
    for rec in recommendations["strategy_recommendations"]:
        print(f"   • {rec}")
    
    print(f"   신뢰도: {recommendations['confidence']:.2f}")
    
    # 학습 상태 확인
    status = meta_system.get_learning_status()
    print(f"\n📊 학습 상태: {status['total_experiences']}개 경험, {status['learning_rules']}개 규칙")

if __name__ == "__main__":
    asyncio.run(test_meta_learning_system()) 