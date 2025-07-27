"""
Performance monitoring system for Cherry AI Streamlit Platform.
Tracks system metrics, response times, memory usage, and performance targets.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging
import gc
import sys

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    response_time: float
    active_sessions: int
    agent_response_times: Dict[str, float] = field(default_factory=dict)
    file_processing_time: Optional[float] = None
    file_size_mb: Optional[float] = None
    concurrent_requests: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0

@dataclass
class PerformanceTarget:
    """Performance target configuration"""
    max_file_processing_time: float = 10.0  # seconds for 10MB files
    max_memory_per_session: float = 1024.0  # MB
    max_response_time: float = 5.0  # seconds
    max_concurrent_users: int = 50
    min_cache_hit_rate: float = 0.7  # 70%
    max_error_rate: float = 0.05  # 5%

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system with real-time metrics collection,
    alerting, and optimization recommendations.
    """
    
    def __init__(self, targets: Optional[PerformanceTarget] = None):
        self.targets = targets or PerformanceTarget()
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
        self.agent_metrics: Dict[str, deque] = {}
        self.file_processing_metrics: deque = deque(maxlen=100)
        
        # Performance alerts
        self.alert_callbacks: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 5.0  # seconds
        
        # Cache metrics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize agent metrics storage
        for port in range(8306, 8316):
            self.agent_metrics[str(port)] = deque(maxlen=100)
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check for performance issues
                self._check_performance_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024 * 1024 * 1024)  # GB
        
        # Calculate average response times
        avg_response_time = self._calculate_average_response_time()
        
        # Agent response times
        agent_response_times = {}
        for agent_id, times in self.agent_metrics.items():
            if times:
                agent_response_times[agent_id] = sum(times) / len(times)
        
        # Cache hit rate
        cache_hit_rate = 0.0
        if self.cache_stats['total_requests'] > 0:
            cache_hit_rate = self.cache_stats['hits'] / self.cache_stats['total_requests']
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            response_time=avg_response_time,
            active_sessions=len(self.session_metrics),
            agent_response_times=agent_response_times,
            concurrent_requests=self._count_concurrent_requests(),
            cache_hit_rate=cache_hit_rate,
            error_rate=self._calculate_error_rate()
        )
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from recent metrics"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_times = [m.response_time for m in list(self.metrics_history)[-10:] if m.response_time > 0]
        return sum(recent_times) / len(recent_times) if recent_times else 0.0
    
    def _count_concurrent_requests(self) -> int:
        """Count current concurrent requests"""
        current_time = datetime.now()
        active_count = 0
        
        for session_id, session_data in self.session_metrics.items():
            last_activity = session_data.get('last_activity', current_time)
            if (current_time - last_activity).total_seconds() < 30:  # Active within 30 seconds
                active_count += 1
        
        return active_count
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if len(self.metrics_history) < 10:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-10:]
        total_requests = sum(m.concurrent_requests for m in recent_metrics)
        
        if total_requests == 0:
            return 0.0
        
        # This would need to be integrated with actual error tracking
        return 0.0  # Placeholder
    
    def record_agent_response_time(self, agent_id: str, response_time: float):
        """Record agent response time"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = deque(maxlen=100)
        
        self.agent_metrics[agent_id].append(response_time)
    
    def record_file_processing(self, file_size_mb: float, processing_time: float):
        """Record file processing metrics"""
        metric = {
            'timestamp': datetime.now(),
            'file_size_mb': file_size_mb,
            'processing_time': processing_time,
            'processing_rate': file_size_mb / processing_time if processing_time > 0 else 0
        }
        
        self.file_processing_metrics.append(metric)
        
        # Check if processing time exceeds target
        if processing_time > self.targets.max_file_processing_time:
            self._trigger_alert(
                "file_processing_slow",
                f"File processing took {processing_time:.2f}s (target: {self.targets.max_file_processing_time}s)",
                {"file_size_mb": file_size_mb, "processing_time": processing_time}
            )
    
    def record_session_activity(self, session_id: str, activity_type: str, metadata: Dict[str, Any] = None):
        """Record session activity"""
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = {
                'created': datetime.now(),
                'activities': [],
                'memory_usage': 0.0,
                'last_activity': datetime.now()
            }
        
        self.session_metrics[session_id]['activities'].append({
            'timestamp': datetime.now(),
            'type': activity_type,
            'metadata': metadata or {}
        })
        self.session_metrics[session_id]['last_activity'] = datetime.now()
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss"""
        self.cache_stats['total_requests'] += 1
        if hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance issues and trigger alerts"""
        
        # CPU usage alert
        if metrics.cpu_usage > 80:
            self._trigger_alert(
                "high_cpu_usage",
                f"CPU usage is {metrics.cpu_usage:.1f}%",
                {"cpu_usage": metrics.cpu_usage}
            )
        
        # Memory usage alert
        if metrics.memory_usage > 85:
            self._trigger_alert(
                "high_memory_usage",
                f"Memory usage is {metrics.memory_usage:.1f}%",
                {"memory_usage": metrics.memory_usage}
            )
        
        # Response time alert
        if metrics.response_time > self.targets.max_response_time:
            self._trigger_alert(
                "slow_response_time",
                f"Response time is {metrics.response_time:.2f}s (target: {self.targets.max_response_time}s)",
                {"response_time": metrics.response_time}
            )
        
        # Concurrent users alert
        if metrics.active_sessions > self.targets.max_concurrent_users:
            self._trigger_alert(
                "high_concurrent_users",
                f"Active sessions: {metrics.active_sessions} (limit: {self.targets.max_concurrent_users})",
                {"active_sessions": metrics.active_sessions}
            )
        
        # Cache hit rate alert
        if metrics.cache_hit_rate < self.targets.min_cache_hit_rate and self.cache_stats['total_requests'] > 100:
            self._trigger_alert(
                "low_cache_hit_rate",
                f"Cache hit rate is {metrics.cache_hit_rate:.1%} (target: {self.targets.min_cache_hit_rate:.1%})",
                {"cache_hit_rate": metrics.cache_hit_rate}
            )
    
    def _trigger_alert(self, alert_type: str, message: str, metadata: Dict[str, Any]):
        """Trigger performance alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'metadata': metadata
        }
        
        self.alert_history.append(alert)
        
        # Keep only recent alerts
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")
        
        self.logger.warning(f"Performance alert [{alert_type}]: {message}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for specified time period"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        max_concurrent = max(m.active_sessions for m in recent_metrics)
        
        # Agent performance
        agent_performance = {}
        for agent_id, times in self.agent_metrics.items():
            if times:
                agent_performance[agent_id] = {
                    'avg_response_time': sum(times) / len(times),
                    'max_response_time': max(times),
                    'min_response_time': min(times),
                    'total_requests': len(times)
                }
        
        # File processing performance
        file_processing_summary = {}
        if self.file_processing_metrics:
            processing_times = [m['processing_time'] for m in self.file_processing_metrics]
            file_sizes = [m['file_size_mb'] for m in self.file_processing_metrics]
            
            file_processing_summary = {
                'avg_processing_time': sum(processing_times) / len(processing_times),
                'max_processing_time': max(processing_times),
                'avg_file_size': sum(file_sizes) / len(file_sizes),
                'total_files_processed': len(self.file_processing_metrics)
            }
        
        return {
            'time_period_minutes': minutes,
            'metrics_count': len(recent_metrics),
            'system_performance': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'avg_response_time': avg_response_time,
                'max_concurrent_sessions': max_concurrent
            },
            'agent_performance': agent_performance,
            'file_processing': file_processing_summary,
            'cache_performance': {
                'hit_rate': self.cache_stats['hits'] / self.cache_stats['total_requests'] if self.cache_stats['total_requests'] > 0 else 0,
                'total_requests': self.cache_stats['total_requests'],
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses']
            },
            'recent_alerts': [a for a in self.alert_history if a['timestamp'] >= cutoff_time],
            'performance_targets': {
                'max_file_processing_time': self.targets.max_file_processing_time,
                'max_memory_per_session': self.targets.max_memory_per_session,
                'max_response_time': self.targets.max_response_time,
                'max_concurrent_users': self.targets.max_concurrent_users
            }
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on current metrics"""
        recommendations = []
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return recommendations
        
        # High CPU usage recommendations
        if current_metrics.cpu_usage > 70:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high' if current_metrics.cpu_usage > 85 else 'medium',
                'title': 'High CPU Usage Detected',
                'description': f'CPU usage is at {current_metrics.cpu_usage:.1f}%',
                'recommendations': [
                    'Enable request queuing to limit concurrent processing',
                    'Implement result caching for frequently requested analyses',
                    'Consider horizontal scaling with additional agent instances',
                    'Optimize data processing algorithms for better efficiency'
                ]
            })
        
        # High memory usage recommendations
        if current_metrics.memory_usage > 75:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high' if current_metrics.memory_usage > 90 else 'medium',
                'title': 'High Memory Usage Detected',
                'description': f'Memory usage is at {current_metrics.memory_usage:.1f}%',
                'recommendations': [
                    'Implement data streaming for large file processing',
                    'Enable garbage collection optimization',
                    'Limit maximum file size for uploads',
                    'Implement session cleanup for inactive users',
                    'Use memory-efficient data structures'
                ]
            })
        
        # Slow response time recommendations
        if current_metrics.response_time > self.targets.max_response_time:
            recommendations.append({
                'type': 'response_time_optimization',
                'priority': 'high',
                'title': 'Slow Response Times',
                'description': f'Average response time is {current_metrics.response_time:.2f}s',
                'recommendations': [
                    'Implement response caching for common queries',
                    'Optimize database queries and data access patterns',
                    'Enable parallel processing for independent operations',
                    'Implement request prioritization',
                    'Consider using faster storage solutions'
                ]
            })
        
        # Low cache hit rate recommendations
        if current_metrics.cache_hit_rate < self.targets.min_cache_hit_rate and self.cache_stats['total_requests'] > 50:
            recommendations.append({
                'type': 'cache_optimization',
                'priority': 'medium',
                'title': 'Low Cache Hit Rate',
                'description': f'Cache hit rate is {current_metrics.cache_hit_rate:.1%}',
                'recommendations': [
                    'Increase cache size and retention time',
                    'Implement smarter cache key strategies',
                    'Pre-cache common analysis results',
                    'Implement cache warming for popular datasets',
                    'Review cache eviction policies'
                ]
            })
        
        # Agent performance recommendations
        slow_agents = []
        for agent_id, avg_time in current_metrics.agent_response_times.items():
            if avg_time > 10.0:  # 10 second threshold
                slow_agents.append((agent_id, avg_time))
        
        if slow_agents:
            recommendations.append({
                'type': 'agent_optimization',
                'priority': 'medium',
                'title': 'Slow Agent Performance',
                'description': f'{len(slow_agents)} agents showing slow response times',
                'recommendations': [
                    'Review agent resource allocation',
                    'Implement agent-specific caching',
                    'Consider agent load balancing',
                    'Optimize agent communication protocols',
                    'Monitor agent health and restart if needed'
                ],
                'affected_agents': slow_agents
            })
        
        return recommendations
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        sessions_to_remove = []
        for session_id, session_data in self.session_metrics.items():
            if session_data['last_activity'] < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.session_metrics[session_id]
        
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
        
        # Force garbage collection
        gc.collect()
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        summary = self.get_metrics_summary()
        
        if format.lower() == 'json':
            return json.dumps(summary, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()