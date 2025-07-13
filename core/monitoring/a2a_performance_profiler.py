"""
A2A 브로커 성능 프로파일링 도구
Phase 2.1: A2A 통신 최적화를 위한 성능 분석

기능:
- 메시지 라우팅 성능 측정
- 응답 시간 분석
- 메모리 사용 패턴 추적
- 병목점 식별
- 연결 상태 모니터링
"""

import asyncio
import time
import psutil
import statistics
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import httpx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """성능 메트릭 데이터 클래스"""
    timestamp: datetime
    endpoint: str
    response_time: float
    status_code: int
    memory_usage_mb: float
    cpu_usage_percent: float
    error: Optional[str] = None

@dataclass
class BottleneckAnalysis:
    """병목점 분석 결과"""
    component: str
    severity: str  # "low", "medium", "high", "critical"
    avg_response_time: float
    max_response_time: float
    error_rate: float
    memory_pressure: float
    cpu_pressure: float
    recommendations: List[str]

@dataclass
class A2AAgentStatus:
    """A2A 에이전트 상태"""
    name: str
    port: int
    url: str
    status: str  # "healthy", "degraded", "unhealthy", "unreachable"
    avg_response_time: float
    error_rate: float
    last_check: datetime
    agent_card_accessible: bool
    memory_usage: float
    cpu_usage: float

class A2APerformanceProfiler:
    """A2A 브로커 성능 프로파일링 시스템"""
    
    def __init__(self, monitoring_interval: int = 5):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.agent_statuses: Dict[str, A2AAgentStatus] = {}
        self.bottlenecks: List[BottleneckAnalysis] = []
        
        # A2A 에이전트 구성
        self.a2a_agents = {
            "orchestrator": {"port": 8100, "url": "http://localhost:8100"},
            "data_cleaning": {"port": 8306, "url": "http://localhost:8306"},
            "data_loader": {"port": 8307, "url": "http://localhost:8307"},
            "data_visualization": {"port": 8308, "url": "http://localhost:8308"},
            "data_wrangling": {"port": 8309, "url": "http://localhost:8309"},
            "feature_engineering": {"port": 8310, "url": "http://localhost:8310"},
            "sql_database": {"port": 8311, "url": "http://localhost:8311"},
            "eda_tools": {"port": 8312, "url": "http://localhost:8312"},
            "h2o_modeling": {"port": 8313, "url": "http://localhost:8313"},
            "mlflow_tracking": {"port": 8314, "url": "http://localhost:8314"},
            "pandas_data_analyst": {"port": 8315, "url": "http://localhost:8315"}
        }
        
        # 성능 추적용 타이머
        self.request_timers: Dict[str, float] = {}
        self.memory_samples: deque = deque(maxlen=100)
        self.cpu_samples: deque = deque(maxlen=100)
        
        # 결과 저장 경로
        self.results_dir = Path("monitoring/performance_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def profile_message_routing_performance(self, test_messages: List[str], iterations: int = 10) -> Dict[str, Any]:
        """메시지 라우팅 성능 프로파일링"""
        logger.info(f"🔍 메시지 라우팅 성능 프로파일링 시작 (반복: {iterations}회)")
        
        routing_results = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for agent_name, agent_config in self.a2a_agents.items():
                logger.info(f"🧪 {agent_name} 성능 테스트 중...")
                
                agent_metrics = []
                
                for iteration in range(iterations):
                    for message in test_messages:
                        start_time = time.time()
                        memory_before = self._get_memory_usage()
                        cpu_before = self._get_cpu_usage()
                        
                        try:
                            # A2A 메시지 전송
                            payload = {
                                "jsonrpc": "2.0",
                                "id": f"perf_test_{iteration}",
                                "method": "message/send",
                                "params": {
                                    "message": {
                                        "role": "user",
                                        "parts": [{"kind": "text", "text": message}],
                                        "messageId": f"perf_msg_{time.time()}"
                                    },
                                    "metadata": {}
                                }
                            }
                            
                            response = await client.post(
                                agent_config["url"],
                                json=payload,
                                headers={"Content-Type": "application/json"}
                            )
                            
                            end_time = time.time()
                            memory_after = self._get_memory_usage()
                            cpu_after = self._get_cpu_usage()
                            
                            response_time = end_time - start_time
                            memory_delta = memory_after - memory_before
                            cpu_delta = cpu_after - cpu_before
                            
                            metric = PerformanceMetric(
                                timestamp=datetime.now(),
                                endpoint=agent_config["url"],
                                response_time=response_time,
                                status_code=response.status_code,
                                memory_usage_mb=memory_delta,
                                cpu_usage_percent=cpu_delta,
                                error=None if response.status_code == 200 else f"HTTP {response.status_code}"
                            )
                            
                            agent_metrics.append(metric)
                            
                        except Exception as e:
                            end_time = time.time()
                            response_time = end_time - start_time
                            
                            error_metric = PerformanceMetric(
                                timestamp=datetime.now(),
                                endpoint=agent_config["url"],
                                response_time=response_time,
                                status_code=0,
                                memory_usage_mb=0,
                                cpu_usage_percent=0,
                                error=str(e)
                            )
                            
                            agent_metrics.append(error_metric)
                
                # 에이전트별 성능 분석
                routing_results[agent_name] = self._analyze_agent_metrics(agent_metrics)
        
        # 전체 라우팅 성능 분석
        overall_analysis = self._analyze_overall_routing_performance(routing_results)
        
        # 결과 저장
        await self._save_performance_results(routing_results, overall_analysis)
        
        return {
            "agent_results": routing_results,
            "overall_analysis": overall_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_agent_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """개별 에이전트 메트릭 분석"""
        if not metrics:
            return {"status": "no_data", "error": "No metrics available"}
        
        successful_metrics = [m for m in metrics if m.error is None]
        failed_metrics = [m for m in metrics if m.error is not None]
        
        if not successful_metrics:
            return {
                "status": "all_failed",
                "total_requests": len(metrics),
                "failed_requests": len(failed_metrics),
                "error_rate": 1.0,
                "common_errors": [m.error for m in failed_metrics[:5]]
            }
        
        response_times = [m.response_time for m in successful_metrics]
        memory_usage = [m.memory_usage_mb for m in successful_metrics]
        cpu_usage = [m.cpu_usage_percent for m in successful_metrics]
        
        analysis = {
            "status": "healthy" if len(failed_metrics) == 0 else "degraded",
            "total_requests": len(metrics),
            "successful_requests": len(successful_metrics),
            "failed_requests": len(failed_metrics),
            "error_rate": len(failed_metrics) / len(metrics),
            "response_times": {
                "avg": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99)
            },
            "memory_usage": {
                "avg_delta_mb": statistics.mean(memory_usage),
                "max_delta_mb": max(memory_usage) if memory_usage else 0,
                "min_delta_mb": min(memory_usage) if memory_usage else 0
            },
            "cpu_usage": {
                "avg_delta_percent": statistics.mean(cpu_usage),
                "max_delta_percent": max(cpu_usage) if cpu_usage else 0
            }
        }
        
        # 성능 등급 평가
        analysis["performance_grade"] = self._calculate_performance_grade(analysis)
        
        return analysis
    
    def _analyze_overall_routing_performance(self, routing_results: Dict[str, Any]) -> Dict[str, Any]:
        """전체 라우팅 성능 분석"""
        healthy_agents = []
        degraded_agents = []
        failed_agents = []
        
        total_response_times = []
        total_error_rates = []
        
        for agent_name, results in routing_results.items():
            status = results.get("status", "unknown")
            
            if status == "healthy":
                healthy_agents.append(agent_name)
            elif status == "degraded":
                degraded_agents.append(agent_name)
            else:
                failed_agents.append(agent_name)
            
            if "response_times" in results:
                total_response_times.append(results["response_times"]["avg"])
            
            if "error_rate" in results:
                total_error_rates.append(results["error_rate"])
        
        # 전체 시스템 상태 평가
        system_health = "healthy"
        if len(failed_agents) > 3:
            system_health = "critical"
        elif len(failed_agents) > 1 or len(degraded_agents) > 4:
            system_health = "degraded"
        elif len(degraded_agents) > 0:
            system_health = "warning"
        
        overall_analysis = {
            "system_health": system_health,
            "agent_status_summary": {
                "healthy": len(healthy_agents),
                "degraded": len(degraded_agents),
                "failed": len(failed_agents),
                "total": len(routing_results)
            },
            "healthy_agents": healthy_agents,
            "degraded_agents": degraded_agents,
            "failed_agents": failed_agents,
            "performance_summary": {
                "avg_response_time": statistics.mean(total_response_times) if total_response_times else 0,
                "avg_error_rate": statistics.mean(total_error_rates) if total_error_rates else 0,
                "fastest_agent": min(routing_results.items(), key=lambda x: x[1].get("response_times", {}).get("avg", float('inf')))[0] if total_response_times else None,
                "slowest_agent": max(routing_results.items(), key=lambda x: x[1].get("response_times", {}).get("avg", 0))[0] if total_response_times else None
            }
        }
        
        # 병목점 식별
        bottlenecks = self._identify_bottlenecks(routing_results)
        overall_analysis["bottlenecks"] = bottlenecks
        
        # 최적화 권장사항
        recommendations = self._generate_optimization_recommendations(overall_analysis, routing_results)
        overall_analysis["optimization_recommendations"] = recommendations
        
        return overall_analysis
    
    def _identify_bottlenecks(self, routing_results: Dict[str, Any]) -> List[BottleneckAnalysis]:
        """병목점 식별"""
        bottlenecks = []
        
        for agent_name, results in routing_results.items():
            if results.get("status") == "all_failed":
                bottleneck = BottleneckAnalysis(
                    component=agent_name,
                    severity="critical",
                    avg_response_time=0,
                    max_response_time=0,
                    error_rate=1.0,
                    memory_pressure=0,
                    cpu_pressure=0,
                    recommendations=[
                        f"{agent_name} 서버가 응답하지 않습니다. 서버 상태를 확인하세요.",
                        "포트 충돌이나 프로세스 상태를 점검하세요.",
                        "로그를 확인하여 구체적인 오류 원인을 파악하세요."
                    ]
                )
                bottlenecks.append(bottleneck)
                continue
            
            response_times = results.get("response_times", {})
            avg_response_time = response_times.get("avg", 0)
            max_response_time = response_times.get("max", 0)
            error_rate = results.get("error_rate", 0)
            
            # 성능 임계값 기준 평가
            severity = "low"
            recommendations = []
            
            if avg_response_time > 10.0:  # 10초 이상
                severity = "critical"
                recommendations.append(f"{agent_name}의 평균 응답시간이 {avg_response_time:.2f}초로 매우 느립니다.")
                recommendations.append("코드 최적화나 인프라 업그레이드를 고려하세요.")
            elif avg_response_time > 5.0:  # 5초 이상
                severity = "high"
                recommendations.append(f"{agent_name}의 응답시간 최적화가 필요합니다.")
            elif avg_response_time > 2.0:  # 2초 이상
                severity = "medium"
                recommendations.append(f"{agent_name}의 성능 모니터링을 강화하세요.")
            
            if error_rate > 0.1:  # 10% 이상
                if severity == "low":
                    severity = "high"
                recommendations.append(f"{agent_name}의 에러율이 {error_rate*100:.1f}%로 높습니다.")
                recommendations.append("에러 로그를 분석하여 근본 원인을 파악하세요.")
            
            if severity != "low" or recommendations:
                bottleneck = BottleneckAnalysis(
                    component=agent_name,
                    severity=severity,
                    avg_response_time=avg_response_time,
                    max_response_time=max_response_time,
                    error_rate=error_rate,
                    memory_pressure=results.get("memory_usage", {}).get("avg_delta_mb", 0),
                    cpu_pressure=results.get("cpu_usage", {}).get("avg_delta_percent", 0),
                    recommendations=recommendations
                )
                bottlenecks.append(bottleneck)
        
        return sorted(bottlenecks, key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.severity], reverse=True)
    
    def _generate_optimization_recommendations(self, overall_analysis: Dict[str, Any], routing_results: Dict[str, Any]) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        system_health = overall_analysis["system_health"]
        failed_agents = overall_analysis["failed_agents"]
        degraded_agents = overall_analysis["degraded_agents"]
        
        # 시스템 전체 권장사항
        if system_health == "critical":
            recommendations.append("🚨 시스템이 위험한 상태입니다. 즉시 조치가 필요합니다.")
            recommendations.append("실패한 에이전트들을 우선 복구하세요: " + ", ".join(failed_agents))
        elif system_health == "degraded":
            recommendations.append("⚠️ 시스템 성능이 저하되었습니다.")
            if degraded_agents:
                recommendations.append("성능 저하 에이전트 최적화: " + ", ".join(degraded_agents))
        
        # 성능 기반 권장사항
        avg_response_time = overall_analysis["performance_summary"]["avg_response_time"]
        if avg_response_time > 3.0:
            recommendations.append(f"전체 평균 응답시간({avg_response_time:.2f}초)이 목표치(3초)를 초과합니다.")
            recommendations.append("연결 풀 크기 증가를 고려하세요.")
            recommendations.append("타임아웃 설정을 조정하세요.")
        
        # 에이전트별 세부 권장사항
        slowest_agent = overall_analysis["performance_summary"]["slowest_agent"]
        if slowest_agent:
            slowest_time = routing_results[slowest_agent]["response_times"]["avg"]
            recommendations.append(f"가장 느린 에이전트 {slowest_agent}({slowest_time:.2f}초) 우선 최적화")
        
        # 메모리/CPU 권장사항
        high_memory_agents = []
        high_cpu_agents = []
        
        for agent_name, results in routing_results.items():
            memory_usage = results.get("memory_usage", {}).get("avg_delta_mb", 0)
            cpu_usage = results.get("cpu_usage", {}).get("avg_delta_percent", 0)
            
            if memory_usage > 100:  # 100MB 이상
                high_memory_agents.append(f"{agent_name}({memory_usage:.1f}MB)")
            
            if cpu_usage > 20:  # 20% 이상
                high_cpu_agents.append(f"{agent_name}({cpu_usage:.1f}%)")
        
        if high_memory_agents:
            recommendations.append("메모리 사용량이 높은 에이전트: " + ", ".join(high_memory_agents))
            recommendations.append("메모리 최적화 또는 인스턴스 스케일링을 고려하세요.")
        
        if high_cpu_agents:
            recommendations.append("CPU 사용량이 높은 에이전트: " + ", ".join(high_cpu_agents))
            recommendations.append("비동기 처리 최적화를 고려하세요.")
        
        return recommendations
    
    async def continuous_monitoring(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """지속적인 성능 모니터링"""
        logger.info(f"🔄 {duration_minutes}분간 지속적인 성능 모니터링 시작")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        monitoring_data = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "samples": [],
            "alerts": []
        }
        
        while datetime.now() < end_time:
            try:
                # 시스템 리소스 수집
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # 각 에이전트 상태 체크
                agent_statuses = await self._check_all_agents_health()
                
                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "system_memory_percent": memory_info.percent,
                    "system_cpu_percent": cpu_percent,
                    "agent_statuses": agent_statuses
                }
                
                monitoring_data["samples"].append(sample)
                
                # 알람 조건 체크
                alerts = self._check_performance_alerts(sample)
                monitoring_data["alerts"].extend(alerts)
                
                # 모니터링 간격 대기
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                monitoring_data["alerts"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "monitoring_error",
                    "message": str(e)
                })
        
        # 모니터링 결과 분석
        analysis = self._analyze_monitoring_data(monitoring_data)
        monitoring_data["analysis"] = analysis
        
        # 결과 저장
        await self._save_monitoring_results(monitoring_data)
        
        return monitoring_data
    
    async def _check_all_agents_health(self) -> Dict[str, Dict[str, Any]]:
        """모든 에이전트의 헬스 체크"""
        agent_statuses = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for agent_name, agent_config in self.a2a_agents.items():
                try:
                    start_time = time.time()
                    
                    # Agent Card 접근 테스트
                    card_response = await client.get(f"{agent_config['url']}/.well-known/agent.json")
                    card_accessible = card_response.status_code == 200
                    
                    # 헬스 체크 (간단한 메시지)
                    health_payload = {
                        "jsonrpc": "2.0",
                        "id": "health_check",
                        "method": "message/send",
                        "params": {
                            "message": {
                                "role": "user",
                                "parts": [{"kind": "text", "text": "ping"}],
                                "messageId": f"health_{time.time()}"
                            },
                            "metadata": {}
                        }
                    }
                    
                    health_response = await client.post(
                        agent_config["url"],
                        json=health_payload
                    )
                    
                    response_time = time.time() - start_time
                    
                    if health_response.status_code == 200:
                        status = "healthy"
                    else:
                        status = "degraded"
                    
                    agent_statuses[agent_name] = {
                        "status": status,
                        "response_time": response_time,
                        "agent_card_accessible": card_accessible,
                        "status_code": health_response.status_code,
                        "error": None
                    }
                    
                except Exception as e:
                    agent_statuses[agent_name] = {
                        "status": "unreachable",
                        "response_time": 0,
                        "agent_card_accessible": False,
                        "status_code": 0,
                        "error": str(e)
                    }
        
        return agent_statuses
    
    def _check_performance_alerts(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """성능 알람 조건 체크"""
        alerts = []
        timestamp = sample["timestamp"]
        
        # 시스템 리소스 알람
        if sample["system_memory_percent"] > 90:
            alerts.append({
                "timestamp": timestamp,
                "type": "system_memory_critical",
                "message": f"시스템 메모리 사용량이 {sample['system_memory_percent']:.1f}%로 위험 수준입니다."
            })
        elif sample["system_memory_percent"] > 80:
            alerts.append({
                "timestamp": timestamp,
                "type": "system_memory_warning",
                "message": f"시스템 메모리 사용량이 {sample['system_memory_percent']:.1f}%로 높습니다."
            })
        
        if sample["system_cpu_percent"] > 90:
            alerts.append({
                "timestamp": timestamp,
                "type": "system_cpu_critical",
                "message": f"시스템 CPU 사용량이 {sample['system_cpu_percent']:.1f}%로 위험 수준입니다."
            })
        
        # 에이전트별 알람
        for agent_name, agent_status in sample["agent_statuses"].items():
            if agent_status["status"] == "unreachable":
                alerts.append({
                    "timestamp": timestamp,
                    "type": "agent_unreachable",
                    "message": f"에이전트 {agent_name}에 연결할 수 없습니다.",
                    "agent": agent_name
                })
            elif agent_status["response_time"] > 10.0:
                alerts.append({
                    "timestamp": timestamp,
                    "type": "agent_slow_response",
                    "message": f"에이전트 {agent_name}의 응답시간이 {agent_status['response_time']:.2f}초로 느립니다.",
                    "agent": agent_name
                })
        
        return alerts
    
    def _analyze_monitoring_data(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """모니터링 데이터 분석"""
        samples = monitoring_data["samples"]
        
        if not samples:
            return {"error": "No monitoring data available"}
        
        # 시스템 리소스 통계
        memory_usage = [s["system_memory_percent"] for s in samples]
        cpu_usage = [s["system_cpu_percent"] for s in samples]
        
        # 에이전트 가용성 분석
        agent_availability = defaultdict(list)
        agent_response_times = defaultdict(list)
        
        for sample in samples:
            for agent_name, status in sample["agent_statuses"].items():
                agent_availability[agent_name].append(status["status"] == "healthy")
                if status["response_time"] > 0:
                    agent_response_times[agent_name].append(status["response_time"])
        
        analysis = {
            "monitoring_duration_minutes": len(samples) * self.monitoring_interval / 60,
            "total_samples": len(samples),
            "system_resources": {
                "memory": {
                    "avg_percent": statistics.mean(memory_usage),
                    "max_percent": max(memory_usage),
                    "min_percent": min(memory_usage)
                },
                "cpu": {
                    "avg_percent": statistics.mean(cpu_usage),
                    "max_percent": max(cpu_usage),
                    "min_percent": min(cpu_usage)
                }
            },
            "agent_availability": {
                agent: {
                    "uptime_percent": sum(statuses) / len(statuses) * 100,
                    "total_checks": len(statuses),
                    "healthy_checks": sum(statuses)
                }
                for agent, statuses in agent_availability.items()
            },
            "agent_performance": {
                agent: {
                    "avg_response_time": statistics.mean(times),
                    "max_response_time": max(times),
                    "min_response_time": min(times)
                }
                for agent, times in agent_response_times.items() if times
            },
            "alerts_summary": {
                "total_alerts": len(monitoring_data["alerts"]),
                "alert_types": list(set(alert["type"] for alert in monitoring_data["alerts"]))
            }
        }
        
        return analysis
    
    async def generate_performance_visualization(self, results: Dict[str, Any]) -> str:
        """성능 분석 결과 시각화"""
        logger.info("📊 성능 분석 결과 시각화 생성 중...")
        
        # 시각화 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('A2A 브로커 성능 분석 결과', fontsize=16, fontweight='bold')
        
        # 1. 에이전트별 응답시간 비교
        agent_names = []
        avg_response_times = []
        error_rates = []
        
        for agent_name, result in results["agent_results"].items():
            if result.get("status") in ["healthy", "degraded"]:
                agent_names.append(agent_name.replace("_", "\n"))
                avg_response_times.append(result["response_times"]["avg"])
                error_rates.append(result["error_rate"] * 100)
        
        if agent_names:
            axes[0, 0].bar(agent_names, avg_response_times, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('에이전트별 평균 응답시간', fontweight='bold')
            axes[0, 0].set_ylabel('응답시간 (초)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 목표 응답시간 라인 추가
            axes[0, 0].axhline(y=3.0, color='red', linestyle='--', label='목표 (3초)')
            axes[0, 0].legend()
        
        # 2. 에이전트별 에러율
        if agent_names:
            bars = axes[0, 1].bar(agent_names, error_rates, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('에이전트별 에러율', fontweight='bold')
            axes[0, 1].set_ylabel('에러율 (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 10% 이상 에러율 강조
            for i, (bar, rate) in enumerate(zip(bars, error_rates)):
                if rate > 10:
                    bar.set_color('red')
        
        # 3. 시스템 상태 요약
        status_summary = results["overall_analysis"]["agent_status_summary"]
        status_labels = list(status_summary.keys())
        status_values = list(status_summary.values())
        colors = ['green', 'orange', 'red', 'blue']
        
        axes[1, 0].pie(status_values, labels=status_labels, colors=colors[:len(status_labels)], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('에이전트 상태 분포', fontweight='bold')
        
        # 4. 병목점 심각도
        bottlenecks = results["overall_analysis"].get("bottlenecks", [])
        if bottlenecks:
            severity_count = defaultdict(int)
            for bottleneck in bottlenecks:
                severity_count[bottleneck.severity] += 1
            
            severity_labels = list(severity_count.keys())
            severity_values = list(severity_count.values())
            severity_colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'green'}
            bar_colors = [severity_colors.get(label, 'gray') for label in severity_labels]
            
            axes[1, 1].bar(severity_labels, severity_values, color=bar_colors, alpha=0.7)
            axes[1, 1].set_title('병목점 심각도 분포', fontweight='bold')
            axes[1, 1].set_ylabel('개수')
        else:
            axes[1, 1].text(0.5, 0.5, '병목점 없음', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('병목점 심각도 분포', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        chart_path = self.results_dir / f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 성능 분석 차트 저장: {chart_path}")
        return str(chart_path)
    
    async def _save_performance_results(self, routing_results: Dict[str, Any], overall_analysis: Dict[str, Any]):
        """성능 분석 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 결과 저장
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "routing_results": routing_results,
            "overall_analysis": overall_analysis
        }
        
        json_path = self.results_dir / f"performance_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 성능 분석 결과 저장: {json_path}")
    
    async def _save_monitoring_results(self, monitoring_data: Dict[str, Any]):
        """모니터링 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_path = self.results_dir / f"monitoring_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 모니터링 결과 저장: {json_path}")
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        return psutil.virtual_memory().used / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """현재 CPU 사용률 (%)"""
        return psutil.cpu_percent()
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """백분위수 계산"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_performance_grade(self, analysis: Dict[str, Any]) -> str:
        """성능 등급 계산"""
        error_rate = analysis.get("error_rate", 0)
        avg_response_time = analysis.get("response_times", {}).get("avg", 0)
        
        if error_rate > 0.2 or avg_response_time > 10:
            return "F"
        elif error_rate > 0.1 or avg_response_time > 5:
            return "D"
        elif error_rate > 0.05 or avg_response_time > 3:
            return "C"
        elif error_rate > 0.01 or avg_response_time > 1.5:
            return "B"
        else:
            return "A"


# 사용 예시 및 테스트
async def main():
    """성능 프로파일링 실행 예시"""
    profiler = A2APerformanceProfiler()
    
    # 테스트 메시지들
    test_messages = [
        "Hello",
        "데이터를 분석해주세요",
        "시각화를 만들어주세요",
        "요약 통계를 계산해주세요"
    ]
    
    # 성능 프로파일링 실행
    results = await profiler.profile_message_routing_performance(test_messages, iterations=5)
    
    # 결과 출력
    print("🔍 A2A 브로커 성능 프로파일링 결과:")
    print(f"시스템 상태: {results['overall_analysis']['system_health']}")
    print(f"평균 응답시간: {results['overall_analysis']['performance_summary']['avg_response_time']:.2f}초")
    
    # 시각화 생성
    chart_path = await profiler.generate_performance_visualization(results)
    print(f"📊 성능 차트: {chart_path}")

if __name__ == "__main__":
    asyncio.run(main()) 