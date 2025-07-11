#!/usr/bin/env python3
"""
📊 Enhanced Log Analyzer for CherryAI Production Environment

향상된 로그 분석 및 패턴 탐지 시스템
- 실시간 로그 파일 모니터링
- 에러 패턴 자동 탐지
- 성능 병목점 분석
- 이상 행동 패턴 감지
- 로그 집계 및 요약
- 자동 로그 로테이션
- 예측적 문제 탐지

Author: CherryAI Production Team
"""

import os
import re
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Pattern
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from pathlib import Path
import gzip
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 우리 시스템 임포트
try:
    from core.integrated_alert_system import get_integrated_alert_system, AlertSeverity, AlertCategory
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """로그 엔트리"""
    timestamp: datetime
    level: str
    source: str
    message: str
    raw_line: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogPattern:
    """로그 패턴 정의"""
    pattern_id: str
    name: str
    regex: Pattern
    severity: AlertSeverity
    category: str
    description: str
    threshold_count: int = 5
    time_window_minutes: int = 5
    enabled: bool = True


@dataclass
class PatternMatch:
    """패턴 매치 결과"""
    pattern_id: str
    pattern_name: str
    matched_entries: List[LogEntry]
    count: int
    first_occurrence: datetime
    last_occurrence: datetime
    severity: AlertSeverity
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogAnalysisReport:
    """로그 분석 보고서"""
    start_time: datetime
    end_time: datetime
    total_entries: int
    entries_by_level: Dict[str, int]
    entries_by_source: Dict[str, int]
    pattern_matches: List[PatternMatch]
    performance_metrics: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]


class LogFileWatcher(FileSystemEventHandler):
    """로그 파일 감시자"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        super().__init__()
    
    def on_modified(self, event):
        """파일 수정 이벤트 처리"""
        if not event.is_directory and event.src_path.endswith('.log'):
            self.analyzer._process_log_file_update(event.src_path)


class EnhancedLogAnalyzer:
    """향상된 로그 분석 시스템"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # 로그 엔트리 저장
        self.recent_entries: deque = deque(maxlen=10000)
        self.pattern_matches: Dict[str, List[PatternMatch]] = defaultdict(list)
        
        # 패턴 정의
        self.patterns: Dict[str, LogPattern] = {}
        self._initialize_patterns()
        
        # 파일 감시
        self.observer = Observer()
        self.watcher = LogFileWatcher(self)
        self.monitoring_active = False
        
        # 분석 스레드
        self.analysis_thread = None
        self.analysis_interval = 30  # 30초마다 분석
        
        # 성능 추적
        self.performance_metrics = {
            "processed_lines": 0,
            "processing_time": 0.0,
            "error_count": 0,
            "pattern_matches": 0
        }
        
        # 알림 시스템 연동
        if ALERT_SYSTEM_AVAILABLE:
            self.alert_system = get_integrated_alert_system()
        else:
            self.alert_system = None
        
        # 파일 추적
        self.file_positions: Dict[str, int] = {}
        
        logger.info(f"📊 로그 분석 시스템 초기화 완료 - 디렉토리: {self.log_directory}")
    
    def _initialize_patterns(self):
        """위험한 로그 패턴 초기화"""
        patterns_config = [
            {
                "pattern_id": "error_burst",
                "name": "에러 버스트",
                "regex": r"ERROR|CRITICAL|FATAL",
                "severity": AlertSeverity.HIGH,
                "category": "error",
                "description": "짧은 시간 내 다수의 에러 발생",
                "threshold_count": 10,
                "time_window_minutes": 5
            },
            {
                "pattern_id": "memory_error",
                "name": "메모리 오류",
                "regex": r"MemoryError|OutOfMemoryError|memory|Memory.*failed",
                "severity": AlertSeverity.CRITICAL,
                "category": "system",
                "description": "메모리 관련 오류 감지",
                "threshold_count": 1,
                "time_window_minutes": 1
            },
            {
                "pattern_id": "connection_timeout",
                "name": "연결 타임아웃",
                "regex": r"timeout|TimeoutError|Connection.*timeout|Request.*timeout",
                "severity": AlertSeverity.HIGH,
                "category": "network",
                "description": "네트워크 연결 타임아웃",
                "threshold_count": 5,
                "time_window_minutes": 3
            },
            {
                "pattern_id": "agent_failure",
                "name": "에이전트 실패",
                "regex": r"Agent.*failed|Agent.*error|A2A.*failed|A2A.*error",
                "severity": AlertSeverity.CRITICAL,
                "category": "agent",
                "description": "A2A 에이전트 실행 실패",
                "threshold_count": 3,
                "time_window_minutes": 2
            },
            {
                "pattern_id": "database_error",
                "name": "데이터베이스 오류",
                "regex": r"DatabaseError|Connection.*refused|SQL.*error|database.*connection",
                "severity": AlertSeverity.HIGH,
                "category": "database",
                "description": "데이터베이스 관련 오류",
                "threshold_count": 3,
                "time_window_minutes": 5
            },
            {
                "pattern_id": "performance_degradation",
                "name": "성능 저하",
                "regex": r"slow|performance|optimization|처리.*느림|응답.*지연",
                "severity": AlertSeverity.MEDIUM,
                "category": "performance",
                "description": "시스템 성능 저하 징후",
                "threshold_count": 15,
                "time_window_minutes": 10
            },
            {
                "pattern_id": "security_alert",
                "name": "보안 경고",
                "regex": r"security|Security|unauthorized|Unauthorized|forbidden|Forbidden|attack",
                "severity": AlertSeverity.CRITICAL,
                "category": "security",
                "description": "보안 관련 경고",
                "threshold_count": 1,
                "time_window_minutes": 1
            },
            {
                "pattern_id": "api_error",
                "name": "API 오류",
                "regex": r"API.*error|HTTP.*[45]\d\d|400|401|403|404|500|502|503|504",
                "severity": AlertSeverity.HIGH,
                "category": "api",
                "description": "API 호출 오류",
                "threshold_count": 10,
                "time_window_minutes": 5
            }
        ]
        
        for pattern_config in patterns_config:
            pattern = LogPattern(
                pattern_id=pattern_config["pattern_id"],
                name=pattern_config["name"],
                regex=re.compile(pattern_config["regex"], re.IGNORECASE),
                severity=pattern_config["severity"],
                category=pattern_config["category"],
                description=pattern_config["description"],
                threshold_count=pattern_config["threshold_count"],
                time_window_minutes=pattern_config["time_window_minutes"]
            )
            self.patterns[pattern.pattern_id] = pattern
    
    def start_monitoring(self):
        """로그 모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            
            # 파일 감시 시작
            self.observer.schedule(self.watcher, str(self.log_directory), recursive=True)
            self.observer.start()
            
            # 분석 스레드 시작
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()
            
            # 기존 로그 파일 초기 처리
            self._process_existing_logs()
            
            logger.info("🔍 로그 모니터링 시작")
    
    def stop_monitoring(self):
        """로그 모니터링 중지"""
        self.monitoring_active = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=10)
        
        logger.info("🛑 로그 모니터링 중지")
    
    def _process_existing_logs(self):
        """기존 로그 파일 처리"""
        log_files = list(self.log_directory.glob("*.log"))
        
        for log_file in log_files:
            try:
                self._process_log_file(str(log_file))
            except Exception as e:
                logger.error(f"로그 파일 처리 실패 {log_file}: {e}")
    
    def _process_log_file_update(self, file_path: str):
        """로그 파일 업데이트 처리"""
        try:
            self._process_log_file(file_path, incremental=True)
        except Exception as e:
            logger.error(f"로그 파일 업데이트 처리 실패 {file_path}: {e}")
    
    def _process_log_file(self, file_path: str, incremental: bool = False):
        """로그 파일 처리"""
        start_time = time.time()
        processed_lines = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 증분 처리를 위한 위치 복원
                if incremental and file_path in self.file_positions:
                    f.seek(self.file_positions[file_path])
                
                for line in f:
                    line = line.strip()
                    if line:
                        entry = self._parse_log_line(line, file_path)
                        if entry:
                            self.recent_entries.append(entry)
                            self._check_patterns(entry)
                            processed_lines += 1
                
                # 파일 위치 저장
                self.file_positions[file_path] = f.tell()
        
        except Exception as e:
            logger.error(f"로그 파일 읽기 오류 {file_path}: {e}")
            self.performance_metrics["error_count"] += 1
        
        # 성능 메트릭 업데이트
        processing_time = time.time() - start_time
        self.performance_metrics["processed_lines"] += processed_lines
        self.performance_metrics["processing_time"] += processing_time
        
        if processed_lines > 0:
            logger.debug(f"로그 처리 완료: {file_path} ({processed_lines}줄, {processing_time:.2f}초)")
    
    def _parse_log_line(self, line: str, source_file: str) -> Optional[LogEntry]:
        """로그 라인 파싱"""
        try:
            # 다양한 로그 형식 지원
            patterns = [
                # 표준 Python logging 형식
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)',
                # ISO 시간 형식
                r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\.\d]*Z?) (\w+) (.+)',
                # 간단한 시간 형식
                r'(\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)',
                # 타임스탬프 없는 형식
                r'(\w+): (.+)'
            ]
            
            timestamp = datetime.now()  # 기본값
            level = "INFO"
            message = line
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    
                    if len(groups) >= 4:  # 표준 Python logging
                        timestamp_str, level, logger_name, message = groups
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        except:
                            pass
                    elif len(groups) >= 3:  # ISO 또는 간단한 형식
                        timestamp_str, level, message = groups
                        try:
                            # ISO 형식 시도
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except:
                            try:
                                # 시간만 있는 형식
                                time_part = datetime.strptime(timestamp_str, '%H:%M:%S').time()
                                timestamp = datetime.combine(datetime.now().date(), time_part)
                            except:
                                pass
                    elif len(groups) >= 2:  # 레벨과 메시지만
                        level, message = groups
                    
                    break
            
            return LogEntry(
                timestamp=timestamp,
                level=level.upper(),
                source=Path(source_file).name,
                message=message.strip(),
                raw_line=line
            )
        
        except Exception as e:
            logger.debug(f"로그 라인 파싱 실패: {e}")
            return None
    
    def _check_patterns(self, entry: LogEntry):
        """로그 엔트리에 대한 패턴 체크"""
        for pattern_id, pattern in self.patterns.items():
            if not pattern.enabled:
                continue
            
            if pattern.regex.search(entry.message) or pattern.regex.search(entry.raw_line):
                self._record_pattern_match(pattern_id, entry)
    
    def _record_pattern_match(self, pattern_id: str, entry: LogEntry):
        """패턴 매치 기록"""
        pattern = self.patterns[pattern_id]
        
        # 시간 윈도우 내의 매치들만 유지
        cutoff_time = datetime.now() - timedelta(minutes=pattern.time_window_minutes)
        recent_matches = [
            match for match in self.pattern_matches[pattern_id]
            if match.last_occurrence >= cutoff_time
        ]
        
        # 새로운 매치 추가
        if recent_matches:
            # 기존 매치에 추가
            latest_match = recent_matches[-1]
            latest_match.matched_entries.append(entry)
            latest_match.count += 1
            latest_match.last_occurrence = entry.timestamp
        else:
            # 새로운 매치 생성
            new_match = PatternMatch(
                pattern_id=pattern_id,
                pattern_name=pattern.name,
                matched_entries=[entry],
                count=1,
                first_occurrence=entry.timestamp,
                last_occurrence=entry.timestamp,
                severity=pattern.severity
            )
            recent_matches.append(new_match)
        
        self.pattern_matches[pattern_id] = recent_matches
        self.performance_metrics["pattern_matches"] += 1
        
        # 임계값 체크
        latest_match = recent_matches[-1]
        if latest_match.count >= pattern.threshold_count:
            self._trigger_pattern_alert(pattern, latest_match)
    
    def _trigger_pattern_alert(self, pattern: LogPattern, match: PatternMatch):
        """패턴 기반 알림 트리거"""
        logger.warning(f"🚨 패턴 감지: {pattern.name} ({match.count}회 in {pattern.time_window_minutes}분)")
        
        # 알림 시스템에 전송 (실제 구현에서는 alert_system.create_pattern_alert() 사용)
        if self.alert_system:
            try:
                # 여기서 알림 시스템에 패턴 기반 알림 전송
                pass
            except Exception as e:
                logger.error(f"패턴 알림 전송 실패: {e}")
    
    def _analysis_loop(self):
        """분석 루프"""
        while self.monitoring_active:
            try:
                # 정기적인 분석 수행
                self._perform_periodic_analysis()
                
                # 로그 로테이션 체크
                self._check_log_rotation()
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"분석 루프 오류: {e}")
                time.sleep(60)
    
    def _perform_periodic_analysis(self):
        """정기적인 분석 수행"""
        # 최근 로그 엔트리 분석
        if len(self.recent_entries) < 10:
            return
        
        # 이상 징후 탐지
        anomalies = self._detect_anomalies()
        
        if anomalies:
            logger.info(f"🔍 {len(anomalies)}개 이상 징후 감지")
            for anomaly in anomalies:
                logger.warning(f"  - {anomaly['type']}: {anomaly['description']}")
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """이상 징후 탐지"""
        anomalies = []
        
        # 최근 30분간의 엔트리 분석
        cutoff_time = datetime.now() - timedelta(minutes=30)
        recent_entries = [entry for entry in self.recent_entries if entry.timestamp >= cutoff_time]
        
        if not recent_entries:
            return anomalies
        
        # 에러율 이상 증가
        error_entries = [entry for entry in recent_entries if entry.level in ['ERROR', 'CRITICAL', 'FATAL']]
        error_rate = len(error_entries) / len(recent_entries) * 100
        
        if error_rate > 10:  # 10% 이상 에러율
            anomalies.append({
                "type": "high_error_rate",
                "description": f"높은 에러율 감지: {error_rate:.1f}%",
                "severity": "high",
                "details": {"error_rate": error_rate, "error_count": len(error_entries)}
            })
        
        # 로그 볼륨 급증
        entry_count = len(recent_entries)
        if entry_count > 1000:  # 30분에 1000개 이상
            anomalies.append({
                "type": "high_log_volume",
                "description": f"로그 볼륨 급증: {entry_count}개/30분",
                "severity": "medium",
                "details": {"entry_count": entry_count}
            })
        
        # 특정 소스에서의 과도한 로그
        source_counts = defaultdict(int)
        for entry in recent_entries:
            source_counts[entry.source] += 1
        
        for source, count in source_counts.items():
            if count > 500:  # 한 소스에서 500개 이상
                anomalies.append({
                    "type": "source_log_burst",
                    "description": f"{source}에서 과도한 로그: {count}개",
                    "severity": "medium",
                    "details": {"source": source, "count": count}
                })
        
        return anomalies
    
    def _check_log_rotation(self):
        """로그 로테이션 체크"""
        try:
            for log_file in self.log_directory.glob("*.log"):
                file_size = log_file.stat().st_size
                
                # 파일 크기가 100MB 이상이면 로테이션
                if file_size > 100 * 1024 * 1024:
                    self._rotate_log_file(log_file)
        
        except Exception as e:
            logger.error(f"로그 로테이션 체크 실패: {e}")
    
    def _rotate_log_file(self, log_file: Path):
        """로그 파일 로테이션"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{log_file.stem}_{timestamp}.log.gz"
            rotated_path = log_file.parent / rotated_name
            
            # 압축하여 백업
            with open(log_file, 'rb') as f_in:
                with gzip.open(rotated_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 원본 파일 초기화
            with open(log_file, 'w') as f:
                f.write(f"# Log rotated at {datetime.now().isoformat()}\n")
            
            # 파일 위치 초기화
            if str(log_file) in self.file_positions:
                self.file_positions[str(log_file)] = 0
            
            logger.info(f"📦 로그 파일 로테이션 완료: {log_file} -> {rotated_name}")
        
        except Exception as e:
            logger.error(f"로그 파일 로테이션 실패 {log_file}: {e}")
    
    def generate_analysis_report(self, hours: int = 24) -> LogAnalysisReport:
        """로그 분석 보고서 생성"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 해당 기간의 엔트리 필터링
        period_entries = [
            entry for entry in self.recent_entries
            if start_time <= entry.timestamp <= end_time
        ]
        
        # 레벨별 집계
        entries_by_level = defaultdict(int)
        for entry in period_entries:
            entries_by_level[entry.level] += 1
        
        # 소스별 집계
        entries_by_source = defaultdict(int)
        for entry in period_entries:
            entries_by_source[entry.source] += 1
        
        # 패턴 매치 수집
        recent_pattern_matches = []
        for pattern_id, matches in self.pattern_matches.items():
            for match in matches:
                if start_time <= match.first_occurrence <= end_time:
                    recent_pattern_matches.append(match)
        
        # 권장사항 생성
        recommendations = self._generate_recommendations(
            period_entries, recent_pattern_matches, entries_by_level
        )
        
        return LogAnalysisReport(
            start_time=start_time,
            end_time=end_time,
            total_entries=len(period_entries),
            entries_by_level=dict(entries_by_level),
            entries_by_source=dict(entries_by_source),
            pattern_matches=recent_pattern_matches,
            performance_metrics=self.performance_metrics.copy(),
            anomalies=self._detect_anomalies(),
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, entries: List[LogEntry], 
                                pattern_matches: List[PatternMatch],
                                entries_by_level: Dict[str, int]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        total_entries = len(entries)
        if total_entries == 0:
            return ["로그 데이터가 충분하지 않습니다."]
        
        # 에러율 체크
        error_count = entries_by_level.get('ERROR', 0) + entries_by_level.get('CRITICAL', 0)
        error_rate = (error_count / total_entries) * 100
        
        if error_rate > 5:
            recommendations.append(f"높은 에러율 감지 ({error_rate:.1f}%) - 시스템 점검 필요")
        
        # 패턴 매치 기반 권장사항
        if pattern_matches:
            critical_patterns = [m for m in pattern_matches if m.severity == AlertSeverity.CRITICAL]
            if critical_patterns:
                recommendations.append("심각한 패턴이 감지되었습니다 - 즉시 대응 필요")
        
        # 성능 기반 권장사항
        if self.performance_metrics["error_count"] > 10:
            recommendations.append("로그 처리 중 오류가 빈번합니다 - 로그 형식 확인 필요")
        
        if not recommendations:
            recommendations.append("시스템이 정상적으로 작동 중입니다.")
        
        return recommendations
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """모니터링 상태 반환"""
        return {
            "monitoring_active": self.monitoring_active,
            "recent_entries_count": len(self.recent_entries),
            "patterns_defined": len(self.patterns),
            "active_patterns": sum(1 for p in self.patterns.values() if p.enabled),
            "performance_metrics": self.performance_metrics.copy(),
            "log_files_monitored": len(self.file_positions)
        }


# 싱글톤 인스턴스
_log_analyzer_instance = None

def get_enhanced_log_analyzer() -> EnhancedLogAnalyzer:
    """향상된 로그 분석기 인스턴스 반환"""
    global _log_analyzer_instance
    if _log_analyzer_instance is None:
        _log_analyzer_instance = EnhancedLogAnalyzer()
    return _log_analyzer_instance


if __name__ == "__main__":
    # 테스트 실행
    analyzer = get_enhanced_log_analyzer()
    analyzer.start_monitoring()
    
    try:
        # 테스트 실행
        time.sleep(60)
        
        # 분석 보고서 생성
        report = analyzer.generate_analysis_report(hours=1)
        print(f"\n📊 로그 분석 보고서")
        print(f"총 엔트리: {report.total_entries}개")
        print(f"패턴 매치: {len(report.pattern_matches)}개")
        print(f"이상 징후: {len(report.anomalies)}개")
        
    except KeyboardInterrupt:
        analyzer.stop_monitoring()
        print("로그 분석 시스템 종료") 