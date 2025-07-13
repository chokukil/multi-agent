"""
LLM First 검증 시스템 (LLM First Validator)
Phase 3.4: 실시간 LLM First 원칙 준수 검증 및 보장

핵심 기능:
- 실시간 LLM First 준수도 검증
- 자동 위반 탐지 및 알림
- 품질 보증 메커니즘
- 준수도 점수 실시간 계산
- 수정 제안 및 가이드
- 지속적 모니터링 및 보고
"""

import asyncio
import json
import logging
import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
import sqlite3

# 분석 관련 imports
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class ViolationSeverity(Enum):
    """위반 심각도"""
    CRITICAL = "critical"     # 즉시 수정 필요
    HIGH = "high"            # 높은 우선순위
    MEDIUM = "medium"        # 보통 우선순위
    LOW = "low"              # 낮은 우선순위
    INFORMATIONAL = "informational"  # 정보성

class ComplianceArea(Enum):
    """준수 영역"""
    DATA_ANALYSIS = "data_analysis"
    STRATEGY_SELECTION = "strategy_selection"
    INSIGHT_GENERATION = "insight_generation"
    RECOMMENDATION = "recommendation"
    VISUALIZATION = "visualization"
    ERROR_HANDLING = "error_handling"

class ValidationStatus(Enum):
    """검증 상태"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"

@dataclass
class ComplianceViolation:
    """준수 위반 정보"""
    id: str
    timestamp: datetime
    area: ComplianceArea
    severity: ViolationSeverity
    description: str
    llm_usage_ratio: float
    hardcoded_elements: List[str]
    suggested_fix: str
    impact_score: float
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComplianceReport:
    """준수 보고서"""
    timestamp: datetime
    overall_score: float
    area_scores: Dict[ComplianceArea, float]
    violations: List[ComplianceViolation]
    improvements: List[str]
    trends: Dict[str, float]
    recommendations: List[str]
    next_review_date: datetime

class LLMFirstRuleEngine:
    """LLM First 규칙 엔진"""
    
    def __init__(self):
        # 준수 규칙 정의
        self.compliance_rules = {
            ComplianceArea.DATA_ANALYSIS: {
                "min_llm_ratio": 0.8,  # 80% 이상 LLM 사용
                "max_hardcoded_ratio": 0.2,  # 20% 이하 하드코딩
                "required_llm_steps": ["data_profiling", "pattern_discovery", "insight_generation"],
                "forbidden_patterns": ["fixed_thresholds", "static_rules", "hardcoded_conditions"]
            },
            ComplianceArea.STRATEGY_SELECTION: {
                "min_llm_ratio": 0.9,  # 90% 이상 LLM 기반 선택
                "dynamic_selection": True,
                "context_awareness": True,
                "forbidden_patterns": ["fixed_strategy_mapping", "rule_based_selection"]
            },
            ComplianceArea.INSIGHT_GENERATION: {
                "min_llm_ratio": 0.95,  # 95% 이상 LLM 생성
                "template_usage_limit": 0.1,  # 10% 이하 템플릿 사용
                "required_qualities": ["novelty", "contextual_relevance", "actionability"]
            },
            ComplianceArea.RECOMMENDATION: {
                "min_llm_ratio": 0.85,  # 85% 이상 LLM 기반
                "personalization": True,
                "context_specific": True
            },
            ComplianceArea.VISUALIZATION: {
                "dynamic_chart_selection": True,
                "llm_guided_design": True,
                "context_appropriate": True
            },
            ComplianceArea.ERROR_HANDLING: {
                "intelligent_recovery": True,
                "llm_guided_fallback": True,
                "user_friendly_messages": True
            }
        }
        
        # 위반 패턴 정의
        self.violation_patterns = {
            "hardcoded_values": [
                r"if\s+.*==\s*['\"].*['\"]",  # 하드코딩된 문자열 비교
                r"threshold\s*=\s*\d+\.?\d*",  # 하드코딩된 임계값
                r"mapping\s*=\s*\{.*['\"].*['\"].*\}",  # 하드코딩된 매핑
            ],
            "static_rules": [
                r"if\s+.*\s+in\s+\[.*['\"].*['\"].*\]",  # 정적 리스트 체크
                r"switch.*case",  # switch-case 패턴
                r"rules\s*=\s*\[.*\]",  # 정적 규칙 리스트
            ],
            "template_responses": [
                r"return\s+[f]?['\"].*{.*}.*['\"]",  # 템플릿 응답
                r"message\s*=\s*[f]?['\"].*['\"]\.format",  # 문자열 포맷팅
                r"template\s*=\s*['\"].*['\"]",  # 템플릿 정의
            ]
        }
        
        # 준수도 가중치
        self.area_weights = {
            ComplianceArea.DATA_ANALYSIS: 0.25,
            ComplianceArea.STRATEGY_SELECTION: 0.20,
            ComplianceArea.INSIGHT_GENERATION: 0.20,
            ComplianceArea.RECOMMENDATION: 0.15,
            ComplianceArea.VISUALIZATION: 0.10,
            ComplianceArea.ERROR_HANDLING: 0.10
        }
    
    def evaluate_compliance(self, analysis_data: Dict[str, Any]) -> Dict[ComplianceArea, float]:
        """준수도 평가"""
        area_scores = {}
        
        for area, rules in self.compliance_rules.items():
            area_score = self._evaluate_area_compliance(area, rules, analysis_data)
            area_scores[area] = area_score
        
        return area_scores
    
    def _evaluate_area_compliance(self, area: ComplianceArea, rules: Dict[str, Any], 
                                 analysis_data: Dict[str, Any]) -> float:
        """영역별 준수도 평가"""
        compliance_factors = []
        
        # LLM 사용 비율 체크
        if "min_llm_ratio" in rules:
            llm_ratio = analysis_data.get("llm_usage_ratio", 0.0)
            ratio_score = min(1.0, llm_ratio / rules["min_llm_ratio"])
            compliance_factors.append(ratio_score)
        
        # 하드코딩 비율 체크
        if "max_hardcoded_ratio" in rules:
            hardcoded_ratio = analysis_data.get("hardcoded_ratio", 0.0)
            hardcode_score = max(0.0, 1.0 - (hardcoded_ratio / rules["max_hardcoded_ratio"]))
            compliance_factors.append(hardcode_score)
        
        # 필수 LLM 단계 체크
        if "required_llm_steps" in rules:
            required_steps = set(rules["required_llm_steps"])
            completed_steps = set(analysis_data.get("llm_steps", []))
            step_coverage = len(completed_steps & required_steps) / len(required_steps)
            compliance_factors.append(step_coverage)
        
        # 금지 패턴 체크
        if "forbidden_patterns" in rules:
            forbidden_count = analysis_data.get("forbidden_pattern_count", 0)
            pattern_score = max(0.0, 1.0 - (forbidden_count * 0.2))  # 패턴당 20% 감점
            compliance_factors.append(pattern_score)
        
        # 동적 특성 체크
        if "dynamic_selection" in rules and rules["dynamic_selection"]:
            dynamic_score = 1.0 if analysis_data.get("is_dynamic", False) else 0.5
            compliance_factors.append(dynamic_score)
        
        # 컨텍스트 인식 체크
        if "context_awareness" in rules and rules["context_awareness"]:
            context_score = analysis_data.get("context_awareness_score", 0.5)
            compliance_factors.append(context_score)
        
        return statistics.mean(compliance_factors) if compliance_factors else 0.5
    
    def detect_violations(self, analysis_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """위반 탐지"""
        violations = []
        
        for area, rules in self.compliance_rules.items():
            area_violations = self._detect_area_violations(area, rules, analysis_data)
            violations.extend(area_violations)
        
        return violations
    
    def _detect_area_violations(self, area: ComplianceArea, rules: Dict[str, Any],
                               analysis_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """영역별 위반 탐지"""
        violations = []
        
        # LLM 비율 위반
        if "min_llm_ratio" in rules:
            llm_ratio = analysis_data.get("llm_usage_ratio", 0.0)
            if llm_ratio < rules["min_llm_ratio"]:
                violation = ComplianceViolation(
                    id=f"llm_ratio_{area.value}_{int(time.time())}",
                    timestamp=datetime.now(),
                    area=area,
                    severity=ViolationSeverity.HIGH,
                    description=f"LLM 사용 비율 부족: {llm_ratio:.2%} < {rules['min_llm_ratio']:.2%}",
                    llm_usage_ratio=llm_ratio,
                    hardcoded_elements=[],
                    suggested_fix=f"LLM 사용 비율을 {rules['min_llm_ratio']:.2%} 이상으로 증가",
                    impact_score=0.8,
                    confidence=0.9
                )
                violations.append(violation)
        
        # 하드코딩 비율 위반
        if "max_hardcoded_ratio" in rules:
            hardcoded_ratio = analysis_data.get("hardcoded_ratio", 0.0)
            if hardcoded_ratio > rules["max_hardcoded_ratio"]:
                hardcoded_elements = analysis_data.get("hardcoded_elements", [])
                violation = ComplianceViolation(
                    id=f"hardcode_{area.value}_{int(time.time())}",
                    timestamp=datetime.now(),
                    area=area,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"하드코딩 비율 초과: {hardcoded_ratio:.2%} > {rules['max_hardcoded_ratio']:.2%}",
                    llm_usage_ratio=analysis_data.get("llm_usage_ratio", 0.0),
                    hardcoded_elements=hardcoded_elements,
                    suggested_fix="하드코딩된 요소를 LLM 기반 동적 로직으로 대체",
                    impact_score=0.9,
                    confidence=0.95
                )
                violations.append(violation)
        
        # 필수 단계 누락
        if "required_llm_steps" in rules:
            required_steps = set(rules["required_llm_steps"])
            completed_steps = set(analysis_data.get("llm_steps", []))
            missing_steps = required_steps - completed_steps
            
            if missing_steps:
                violation = ComplianceViolation(
                    id=f"missing_steps_{area.value}_{int(time.time())}",
                    timestamp=datetime.now(),
                    area=area,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"필수 LLM 단계 누락: {', '.join(missing_steps)}",
                    llm_usage_ratio=analysis_data.get("llm_usage_ratio", 0.0),
                    hardcoded_elements=[],
                    suggested_fix=f"누락된 LLM 단계 추가: {', '.join(missing_steps)}",
                    impact_score=0.6,
                    confidence=0.8
                )
                violations.append(violation)
        
        return violations

class ComplianceMonitor:
    """준수도 모니터"""
    
    def __init__(self):
        self.rule_engine = LLMFirstRuleEngine()
        self.llm_client = AsyncOpenAI()
        
        # 모니터링 상태
        self.monitoring_active = False
        self.monitoring_interval = 60  # 60초마다 체크
        
        # 알림 설정
        self.alert_thresholds = {
            ViolationSeverity.CRITICAL: 0.0,  # 즉시 알림
            ViolationSeverity.HIGH: 1.0,      # 1분 후 알림
            ViolationSeverity.MEDIUM: 5.0,    # 5분 후 알림
            ViolationSeverity.LOW: 30.0       # 30분 후 알림
        }
        
        # 위반 기록
        self.violation_history: deque = deque(maxlen=1000)
        self.compliance_trends: deque = deque(maxlen=100)
        
        # 콜백 함수들
        self.violation_callbacks: List[Callable] = []
        self.improvement_callbacks: List[Callable] = []
        
        # 데이터베이스 초기화
        self.db_path = Path("core/validation/compliance.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 위반 기록 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    area TEXT,
                    severity TEXT,
                    description TEXT,
                    llm_usage_ratio REAL,
                    hardcoded_elements TEXT,
                    suggested_fix TEXT,
                    impact_score REAL,
                    confidence REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_timestamp DATETIME
                )
            """)
            
            # 준수도 기록 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_history (
                    timestamp DATETIME PRIMARY KEY,
                    overall_score REAL,
                    area_scores TEXT,
                    violations_count INTEGER,
                    improvements_count INTEGER
                )
            """)
            
            conn.commit()
    
    async def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring_active:
            logger.warning("모니터링이 이미 활성화되어 있습니다.")
            return
        
        self.monitoring_active = True
        logger.info("🔍 LLM First 준수도 모니터링 시작")
        
        # 모니터링 루프 시작
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        logger.info("⏹️ LLM First 준수도 모니터링 중지")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 현재 분석 상태 체크
                await self._check_current_compliance()
                
                # 트렌드 분석
                await self._analyze_compliance_trends()
                
                # 대기
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(30)  # 오류 시 30초 대기
    
    async def validate_analysis(self, analysis_data: Dict[str, Any]) -> ComplianceReport:
        """분석 준수도 검증"""
        logger.info("📊 분석 준수도 검증 시작")
        
        # 준수도 평가
        area_scores = self.rule_engine.evaluate_compliance(analysis_data)
        overall_score = self._calculate_overall_score(area_scores)
        
        # 위반 탐지
        violations = self.rule_engine.detect_violations(analysis_data)
        
        # LLM 기반 개선 제안
        improvements = await self._generate_improvements(analysis_data, violations)
        
        # 트렌드 분석
        trends = self._analyze_trends()
        
        # 권장사항 생성
        recommendations = await self._generate_recommendations(area_scores, violations)
        
        # 보고서 생성
        report = ComplianceReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            area_scores=area_scores,
            violations=violations,
            improvements=improvements,
            trends=trends,
            recommendations=recommendations,
            next_review_date=datetime.now() + timedelta(hours=24)
        )
        
        # 기록 저장
        await self._save_compliance_record(report)
        
        # 위반 알림
        await self._handle_violations(violations)
        
        logger.info(f"✅ 준수도 검증 완료: {overall_score:.1f}/100")
        return report
    
    def _calculate_overall_score(self, area_scores: Dict[ComplianceArea, float]) -> float:
        """전체 준수도 점수 계산"""
        weighted_sum = sum(
            score * self.rule_engine.area_weights[area] 
            for area, score in area_scores.items()
        )
        return weighted_sum * 100  # 0-100 스케일
    
    async def _generate_improvements(self, analysis_data: Dict[str, Any], 
                                   violations: List[ComplianceViolation]) -> List[str]:
        """개선 제안 생성 (LLM 기반)"""
        if not violations:
            return ["현재 LLM First 원칙을 잘 준수하고 있습니다."]
        
        # 위반 요약
        violation_summary = self._summarize_violations(violations)
        
        improvement_prompt = f"""
LLM First 원칙 개선 전문가로서 다음 위반 사항에 대한 구체적인 개선 제안을 해주세요.

위반 요약:
{violation_summary}

현재 LLM 사용 비율: {analysis_data.get('llm_usage_ratio', 0):.2%}
하드코딩 비율: {analysis_data.get('hardcoded_ratio', 0):.2%}

다음 형식으로 개선 제안을 해주세요:
1. 즉시 개선 가능한 사항 (3개)
2. 중장기 개선 계획 (2개)
3. LLM 활용도 증대 방안 (2개)

실용적이고 구체적인 제안을 해주세요.
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": improvement_prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            improvement_text = response.choices[0].message.content
            improvements = self._parse_improvements(improvement_text)
            
        except Exception as e:
            logger.warning(f"LLM 개선 제안 생성 실패: {e}")
            improvements = self._fallback_improvements(violations)
        
        return improvements
    
    def _summarize_violations(self, violations: List[ComplianceViolation]) -> str:
        """위반 요약"""
        summary_parts = []
        
        # 심각도별 개수
        severity_counts = defaultdict(int)
        for violation in violations:
            severity_counts[violation.severity] += 1
        
        for severity, count in severity_counts.items():
            summary_parts.append(f"- {severity.value}: {count}개")
        
        # 주요 위반 사항
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            summary_parts.append("\n주요 위반:")
            for violation in critical_violations[:3]:
                summary_parts.append(f"- {violation.description}")
        
        return "\n".join(summary_parts)
    
    def _parse_improvements(self, improvement_text: str) -> List[str]:
        """개선 제안 파싱"""
        improvements = []
        
        lines = improvement_text.split('\n')
        current_improvement = ""
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if line[0].isdigit() or line.startswith('-'):
                    if current_improvement:
                        improvements.append(current_improvement.strip())
                    current_improvement = line
                else:
                    current_improvement += " " + line
        
        if current_improvement:
            improvements.append(current_improvement.strip())
        
        return improvements[:10]  # 최대 10개
    
    def _fallback_improvements(self, violations: List[ComplianceViolation]) -> List[str]:
        """폴백 개선 제안"""
        improvements = []
        
        # 위반 유형별 기본 제안
        if any(v.severity == ViolationSeverity.CRITICAL for v in violations):
            improvements.append("1. 하드코딩된 로직을 LLM 기반 동적 분석으로 즉시 대체")
        
        if any("llm_ratio" in v.id for v in violations):
            improvements.append("2. 분석 단계별 LLM 활용도를 90% 이상으로 증대")
        
        improvements.extend([
            "3. 템플릿 응답을 LLM 생성 응답으로 전환",
            "4. 정적 규칙 기반 로직을 컨텍스트 인식 LLM 판단으로 변경",
            "5. 지속적인 LLM First 원칙 준수 모니터링 강화"
        ])
        
        return improvements
    
    async def _generate_recommendations(self, area_scores: Dict[ComplianceArea, float],
                                      violations: List[ComplianceViolation]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 점수 기반 권장사항
        low_score_areas = [area for area, score in area_scores.items() if score < 0.7]
        
        if low_score_areas:
            recommendations.append(f"우선 개선 영역: {', '.join([area.value for area in low_score_areas])}")
        
        # 위반 기반 권장사항
        critical_count = len([v for v in violations if v.severity == ViolationSeverity.CRITICAL])
        if critical_count > 0:
            recommendations.append(f"긴급 수정 필요: {critical_count}개 심각한 위반 사항")
        
        # 일반적 권장사항
        recommendations.extend([
            "정기적인 LLM First 원칙 교육 및 리뷰 수행",
            "코드 리뷰 시 LLM First 체크리스트 활용",
            "새로운 기능 개발 시 LLM First 설계 가이드 적용"
        ])
        
        return recommendations
    
    def _analyze_trends(self) -> Dict[str, float]:
        """트렌드 분석"""
        if len(self.compliance_trends) < 2:
            return {"trend": 0.0, "confidence": 0.0}
        
        # 최근 점수 변화 분석
        recent_scores = [record["overall_score"] for record in self.compliance_trends]
        
        if len(recent_scores) >= 5:
            early_avg = statistics.mean(recent_scores[:len(recent_scores)//2])
            recent_avg = statistics.mean(recent_scores[len(recent_scores)//2:])
            trend = recent_avg - early_avg
        else:
            trend = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0.0
        
        confidence = min(1.0, len(recent_scores) / 10)  # 10개 기록에서 최대 신뢰도
        
        return {
            "overall_trend": trend,
            "trend_confidence": confidence,
            "improvement_rate": max(0, trend) / 30 if trend > 0 else 0  # 30일 기준 개선률
        }
    
    async def _save_compliance_record(self, report: ComplianceReport):
        """준수도 기록 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 위반 기록 저장
            for violation in report.violations:
                cursor.execute("""
                    INSERT OR REPLACE INTO violations 
                    (id, timestamp, area, severity, description, llm_usage_ratio,
                     hardcoded_elements, suggested_fix, impact_score, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.id,
                    violation.timestamp,
                    violation.area.value,
                    violation.severity.value,
                    violation.description,
                    violation.llm_usage_ratio,
                    json.dumps(violation.hardcoded_elements),
                    violation.suggested_fix,
                    violation.impact_score,
                    violation.confidence
                ))
            
            # 준수도 기록 저장
            cursor.execute("""
                INSERT OR REPLACE INTO compliance_history
                (timestamp, overall_score, area_scores, violations_count, improvements_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                report.timestamp,
                report.overall_score,
                json.dumps({area.value: score for area, score in report.area_scores.items()}),
                len(report.violations),
                len(report.improvements)
            ))
            
            conn.commit()
        
        # 메모리 기록 업데이트
        self.compliance_trends.append({
            "timestamp": report.timestamp,
            "overall_score": report.overall_score,
            "violations_count": len(report.violations)
        })
    
    async def _handle_violations(self, violations: List[ComplianceViolation]):
        """위반 처리"""
        for violation in violations:
            # 심각도에 따른 알림
            await self._send_violation_alert(violation)
            
            # 콜백 실행
            for callback in self.violation_callbacks:
                try:
                    await callback(violation)
                except Exception as e:
                    logger.error(f"위반 콜백 실행 오류: {e}")
    
    async def _send_violation_alert(self, violation: ComplianceViolation):
        """위반 알림 발송"""
        # 심각도에 따른 즉시성 결정
        delay = self.alert_thresholds.get(violation.severity, 0.0)
        
        if delay > 0:
            await asyncio.sleep(delay * 60)  # 분 단위를 초 단위로 변환
        
        logger.warning(f"🚨 LLM First 위반 알림 ({violation.severity.value}): {violation.description}")
        
        # 실제 환경에서는 이메일, Slack 등으로 알림 발송
        alert_message = f"""
LLM First 원칙 위반 발생

위반 ID: {violation.id}
영역: {violation.area.value}
심각도: {violation.severity.value}
설명: {violation.description}
LLM 사용 비율: {violation.llm_usage_ratio:.2%}
제안 수정: {violation.suggested_fix}

즉시 확인이 필요합니다.
"""
        
        # 로그로 알림 (실제로는 외부 시스템 연동)
        logger.info(f"📧 알림 발송: {alert_message}")
    
    async def _check_current_compliance(self):
        """현재 준수도 체크"""
        # 실제 구현에서는 현재 실행 중인 분석의 상태를 체크
        # 여기서는 시뮬레이션
        pass
    
    async def _analyze_compliance_trends(self):
        """준수도 트렌드 분석"""
        if len(self.compliance_trends) >= 5:
            trends = self._analyze_trends()
            
            # 지속적 하락 트렌드 감지
            if trends["overall_trend"] < -5:  # 5점 이상 하락
                logger.warning("📉 LLM First 준수도 하락 트렌드 감지")
                
                # 개선 콜백 실행
                for callback in self.improvement_callbacks:
                    try:
                        await callback(trends)
                    except Exception as e:
                        logger.error(f"개선 콜백 실행 오류: {e}")
    
    def add_violation_callback(self, callback: Callable):
        """위반 콜백 추가"""
        self.violation_callbacks.append(callback)
    
    def add_improvement_callback(self, callback: Callable):
        """개선 콜백 추가"""
        self.improvement_callbacks.append(callback)
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """준수도 대시보드 데이터"""
        current_score = self.compliance_trends[-1]["overall_score"] if self.compliance_trends else 0
        recent_violations = list(self.violation_history)[-10:]
        trends = self._analyze_trends()
        
        return {
            "current_score": current_score,
            "trend": trends["overall_trend"],
            "recent_violations": len(recent_violations),
            "monitoring_active": self.monitoring_active,
            "last_check": datetime.now().isoformat(),
            "historical_data": {
                "scores": [record["overall_score"] for record in self.compliance_trends],
                "timestamps": [record["timestamp"].isoformat() for record in self.compliance_trends]
            }
        }

class LLMFirstQualityAssurance:
    """LLM First 품질 보증"""
    
    def __init__(self):
        self.compliance_monitor = ComplianceMonitor()
        self.quality_gates = {
            "development": 70.0,    # 개발 단계 최소 점수
            "testing": 80.0,        # 테스트 단계 최소 점수
            "production": 90.0      # 프로덕션 배포 최소 점수
        }
        
        # 품질 체크 히스토리
        self.quality_history: deque = deque(maxlen=100)
        
    async def quality_gate_check(self, stage: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """품질 게이트 체크"""
        logger.info(f"🛡️ {stage} 단계 품질 게이트 체크")
        
        # 준수도 검증
        compliance_report = await self.compliance_monitor.validate_analysis(analysis_data)
        
        # 품질 게이트 통과 여부
        minimum_score = self.quality_gates.get(stage, 70.0)
        passed = compliance_report.overall_score >= minimum_score
        
        quality_result = {
            "stage": stage,
            "passed": passed,
            "score": compliance_report.overall_score,
            "minimum_score": minimum_score,
            "violations": len(compliance_report.violations),
            "critical_violations": len([v for v in compliance_report.violations 
                                      if v.severity == ViolationSeverity.CRITICAL]),
            "recommendations": compliance_report.recommendations,
            "next_steps": self._determine_next_steps(passed, compliance_report)
        }
        
        # 히스토리 저장
        self.quality_history.append({
            "timestamp": datetime.now(),
            "stage": stage,
            "result": quality_result
        })
        
        if passed:
            logger.info(f"✅ {stage} 품질 게이트 통과: {compliance_report.overall_score:.1f}/{minimum_score}")
        else:
            logger.warning(f"❌ {stage} 품질 게이트 실패: {compliance_report.overall_score:.1f}/{minimum_score}")
        
        return quality_result
    
    def _determine_next_steps(self, passed: bool, report: ComplianceReport) -> List[str]:
        """다음 단계 결정"""
        if passed:
            return [
                "품질 게이트 통과 - 다음 단계 진행 가능",
                "지속적인 LLM First 원칙 준수 유지",
                "정기적인 준수도 모니터링 수행"
            ]
        else:
            next_steps = ["품질 게이트 실패 - 수정 후 재검토 필요"]
            
            # 심각한 위반이 있는 경우
            critical_violations = [v for v in report.violations if v.severity == ViolationSeverity.CRITICAL]
            if critical_violations:
                next_steps.append(f"우선 수정: {len(critical_violations)}개 심각한 위반 사항")
            
            # 개선 제안 추가
            next_steps.extend(report.improvements[:3])
            
            return next_steps
    
    async def continuous_quality_monitoring(self):
        """지속적 품질 모니터링"""
        await self.compliance_monitor.start_monitoring()
        
        # 품질 개선 콜백 등록
        self.compliance_monitor.add_improvement_callback(self._handle_quality_degradation)
        
    async def _handle_quality_degradation(self, trends: Dict[str, float]):
        """품질 저하 처리"""
        logger.warning("📊 LLM First 품질 저하 감지 - 개선 조치 시작")
        
        # 자동 개선 조치
        improvement_actions = [
            "LLM First 원칙 재교육 스케줄링",
            "코드 리뷰 강화",
            "자동화된 준수도 체크 활성화",
            "개발팀 알림 발송"
        ]
        
        for action in improvement_actions:
            logger.info(f"🔧 개선 조치: {action}")
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """품질 메트릭 조회"""
        if not self.quality_history:
            return {"status": "no_data"}
        
        recent_checks = list(self.quality_history)[-10:]
        
        # 통과율 계산
        passed_count = sum(1 for check in recent_checks if check["result"]["passed"])
        pass_rate = passed_count / len(recent_checks)
        
        # 평균 점수
        avg_score = statistics.mean(check["result"]["score"] for check in recent_checks)
        
        # 트렌드
        scores = [check["result"]["score"] for check in recent_checks]
        trend = scores[-1] - scores[0] if len(scores) > 1 else 0
        
        return {
            "pass_rate": pass_rate,
            "average_score": avg_score,
            "trend": trend,
            "recent_checks": len(recent_checks),
            "quality_gates": self.quality_gates,
            "monitoring_active": self.compliance_monitor.monitoring_active
        }


# 사용 예시 및 테스트
async def test_llm_first_validator():
    """LLM First 검증 시스템 테스트"""
    
    # 품질 보증 시스템 초기화
    qa_system = LLMFirstQualityAssurance()
    
    # 테스트 분석 데이터
    test_analysis_data = {
        "llm_usage_ratio": 0.75,  # 75% LLM 사용
        "hardcoded_ratio": 0.30,  # 30% 하드코딩 (높음)
        "llm_steps": ["data_profiling", "insight_generation"],  # pattern_discovery 누락
        "forbidden_pattern_count": 2,
        "is_dynamic": True,
        "context_awareness_score": 0.8,
        "hardcoded_elements": ["threshold = 0.5", "if status == 'active'"]
    }
    
    print("🔍 LLM First 검증 시스템 테스트 시작...")
    
    # 개발 단계 품질 게이트 체크
    dev_result = await qa_system.quality_gate_check("development", test_analysis_data)
    
    print(f"\n📊 개발 단계 품질 게이트 결과:")
    print(f"   통과 여부: {'✅ 통과' if dev_result['passed'] else '❌ 실패'}")
    print(f"   점수: {dev_result['score']:.1f}/{dev_result['minimum_score']}")
    print(f"   위반 수: {dev_result['violations']}개 (심각: {dev_result['critical_violations']}개)")
    
    print(f"\n💡 주요 권장사항:")
    for i, rec in enumerate(dev_result['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    # 지속적 모니터링 시작
    await qa_system.continuous_quality_monitoring()
    
    print(f"\n📈 품질 메트릭:")
    metrics = qa_system.get_quality_metrics()
    if metrics.get("status") != "no_data":
        print(f"   통과율: {metrics['pass_rate']:.2%}")
        print(f"   평균 점수: {metrics['average_score']:.1f}")
        print(f"   모니터링 활성: {'✅' if metrics['monitoring_active'] else '❌'}")
    
    # 모니터링 중지
    await qa_system.compliance_monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(test_llm_first_validator()) 