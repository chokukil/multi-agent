"""
Concurrent processing system for Cherry AI Streamlit Platform.
Supports multiple agent execution, request queuing, and load balancing.
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
from collections import deque
import weakref

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ConcurrentTask:
    """Task for concurrent execution"""
    id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class WorkerPool:
    """Pool of worker threads/coroutines for concurrent execution"""
    
    def __init__(self, max_workers: int = 10, worker_type: str = "thread"):
        self.max_workers = max_workers
        self.worker_type = worker_type  # "thread" or "async"
        self.active_workers = 0
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the worker pool"""
        if self.running:
            return
        
        self.running = True
        
        # Create worker coroutines
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        self.logger.info(f"Started worker pool with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the worker pool"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        self.logger.info("Worker pool stopped")
    
    async def submit_task(self, task: ConcurrentTask) -> str:
        """Submit a task for execution"""
        await self.task_queue.put(task)
        self.logger.debug(f"Task {task.id} submitted to queue")
        return task.id
    
    async def _worker(self, worker_name: str):
        """Worker coroutine"""
        self.logger.debug(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Execute task
                await self._execute_task(task, worker_name)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {str(e)}")
        
        self.logger.debug(f"Worker {worker_name} stopped")
    
    async def _execute_task(self, task: ConcurrentTask, worker_name: str):
        """Execute a single task"""
        self.active_workers += 1
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        self.logger.debug(f"Worker {worker_name} executing task {task.id}")
        
        try:
            # Set timeout if specified
            if task.timeout_seconds:
                result = await asyncio.wait_for(
                    self._run_task_function(task),
                    timeout=task.timeout_seconds
                )
            else:
                result = await self._run_task_function(task)
            
            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update statistics
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.stats['tasks_processed'] += 1
            self.stats['total_execution_time'] += execution_time
            self.stats['avg_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['tasks_processed']
            )
            
            self.logger.debug(f"Task {task.id} completed in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            task.error = TimeoutError(f"Task timed out after {task.timeout_seconds}s")
            task.status = TaskStatus.FAILED
            self.stats['tasks_failed'] += 1
            self.logger.warning(f"Task {task.id} timed out")
            
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            self.stats['tasks_failed'] += 1
            self.logger.error(f"Task {task.id} failed: {str(e)}")
        
        finally:
            self.active_workers -= 1
    
    async def _run_task_function(self, task: ConcurrentTask) -> Any:
        """Run the task function (sync or async)"""
        if asyncio.iscoroutinefunction(task.function):
            return await task.function(*task.args, **task.kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: task.function(*task.args, **task.kwargs)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics"""
        return {
            'max_workers': self.max_workers,
            'active_workers': self.active_workers,
            'queue_size': self.task_queue.qsize(),
            'running': self.running,
            **self.stats
        }

class TaskScheduler:
    """Priority-based task scheduler"""
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.pending_tasks: Dict[str, ConcurrentTask] = {}
        self.completed_tasks: Dict[str, ConcurrentTask] = {}
        self.priority_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.task_callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the task scheduler"""
        if self.running:
            return
        
        self.running = True
        await self.worker_pool.start()
        
        # Start scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        self.logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler"""
        if not self.running:
            return
        
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        await self.worker_pool.stop()
        self.logger.info("Task scheduler stopped")
    
    async def submit_task(self, 
                         function: Callable,
                         args: tuple = (),
                         kwargs: dict = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout_seconds: Optional[int] = None,
                         session_id: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a task for execution"""
        
        task_id = str(uuid.uuid4())
        task = ConcurrentTask(
            id=task_id,
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Add to pending tasks
        self.pending_tasks[task_id] = task
        
        # Add to priority queue
        self.priority_queues[priority].append(task)
        
        self.logger.debug(f"Task {task_id} scheduled with priority {priority.name}")
        return task_id
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Check for tasks to schedule (highest priority first)
                task_to_schedule = None
                
                for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                    if self.priority_queues[priority]:
                        task_to_schedule = self.priority_queues[priority].popleft()
                        break
                
                if task_to_schedule:
                    # Submit to worker pool
                    await self.worker_pool.submit_task(task_to_schedule)
                    
                    # Move from pending to active tracking
                    if task_to_schedule.id in self.pending_tasks:
                        del self.pending_tasks[task_to_schedule.id]
                
                # Check for completed tasks
                await self._check_completed_tasks()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _check_completed_tasks(self):
        """Check for completed tasks and trigger callbacks"""
        # This would need to be integrated with the worker pool
        # to get notifications of completed tasks
        pass
    
    async def get_task_status(self, task_id: str) -> Optional[ConcurrentTask]:
        """Get task status"""
        if task_id in self.pending_tasks:
            return self.pending_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        if task_id in self.pending_tasks:
            task = self.pending_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            # Remove from priority queue
            for priority_queue in self.priority_queues.values():
                if task in priority_queue:
                    priority_queue.remove(task)
                    break
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.pending_tasks[task_id]
            
            self.logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def add_task_callback(self, task_id: str, callback: Callable):
        """Add callback for task completion"""
        if task_id not in self.task_callbacks:
            self.task_callbacks[task_id] = []
        self.task_callbacks[task_id].append(callback)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        queue_sizes = {}
        for priority, queue in self.priority_queues.items():
            queue_sizes[priority.name] = len(queue)
        
        return {
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'priority_queues': queue_sizes,
            'worker_pool_stats': self.worker_pool.get_stats()
        }

class LoadBalancer:
    """Load balancer for distributing tasks across multiple agents"""
    
    def __init__(self):
        self.agent_loads: Dict[str, int] = {}  # agent_id -> current load
        self.agent_capacities: Dict[str, int] = {}  # agent_id -> max capacity
        self.agent_health: Dict[str, bool] = {}  # agent_id -> health status
        self.load_history: Dict[str, List[int]] = {}  # agent_id -> load history
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent_id: str, capacity: int = 10):
        """Register an agent with the load balancer"""
        self.agent_loads[agent_id] = 0
        self.agent_capacities[agent_id] = capacity
        self.agent_health[agent_id] = True
        self.load_history[agent_id] = []
        self.logger.info(f"Agent {agent_id} registered with capacity {capacity}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_loads:
            del self.agent_loads[agent_id]
            del self.agent_capacities[agent_id]
            del self.agent_health[agent_id]
            del self.load_history[agent_id]
            self.logger.info(f"Agent {agent_id} unregistered")
    
    def update_agent_health(self, agent_id: str, healthy: bool):
        """Update agent health status"""
        if agent_id in self.agent_health:
            self.agent_health[agent_id] = healthy
            if not healthy:
                self.logger.warning(f"Agent {agent_id} marked as unhealthy")
    
    def select_agent(self, strategy: str = "least_loaded") -> Optional[str]:
        """Select best agent based on strategy"""
        healthy_agents = [
            agent_id for agent_id, healthy in self.agent_health.items() 
            if healthy and self.agent_loads[agent_id] < self.agent_capacities[agent_id]
        ]
        
        if not healthy_agents:
            return None
        
        if strategy == "least_loaded":
            return min(healthy_agents, key=lambda a: self.agent_loads[a])
        elif strategy == "round_robin":
            # Simple round-robin (would need state tracking for true round-robin)
            return healthy_agents[0]
        elif strategy == "random":
            import random
            return random.choice(healthy_agents)
        else:
            return healthy_agents[0]
    
    def assign_task(self, agent_id: str) -> bool:
        """Assign a task to an agent"""
        if (agent_id in self.agent_loads and 
            self.agent_health.get(agent_id, False) and
            self.agent_loads[agent_id] < self.agent_capacities[agent_id]):
            
            self.agent_loads[agent_id] += 1
            self._update_load_history(agent_id)
            return True
        
        return False
    
    def complete_task(self, agent_id: str):
        """Mark task as completed for an agent"""
        if agent_id in self.agent_loads and self.agent_loads[agent_id] > 0:
            self.agent_loads[agent_id] -= 1
            self._update_load_history(agent_id)
    
    def _update_load_history(self, agent_id: str):
        """Update load history for an agent"""
        if agent_id in self.load_history:
            self.load_history[agent_id].append(self.agent_loads[agent_id])
            
            # Keep only recent history
            if len(self.load_history[agent_id]) > 100:
                self.load_history[agent_id].pop(0)
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        total_capacity = sum(self.agent_capacities.values())
        total_load = sum(self.agent_loads.values())
        
        agent_stats = {}
        for agent_id in self.agent_loads:
            load = self.agent_loads[agent_id]
            capacity = self.agent_capacities[agent_id]
            utilization = load / capacity if capacity > 0 else 0
            
            agent_stats[agent_id] = {
                'current_load': load,
                'capacity': capacity,
                'utilization': utilization,
                'healthy': self.agent_health[agent_id],
                'avg_load': sum(self.load_history[agent_id]) / len(self.load_history[agent_id]) if self.load_history[agent_id] else 0
            }
        
        return {
            'total_capacity': total_capacity,
            'total_load': total_load,
            'overall_utilization': total_load / total_capacity if total_capacity > 0 else 0,
            'healthy_agents': sum(1 for h in self.agent_health.values() if h),
            'total_agents': len(self.agent_health),
            'agent_stats': agent_stats
        }

class ConcurrentProcessor:
    """Main concurrent processing system"""
    
    def __init__(self, max_workers: int = 20, max_concurrent_users: int = 50):
        self.max_workers = max_workers
        self.max_concurrent_users = max_concurrent_users
        
        # Initialize components
        self.worker_pool = WorkerPool(max_workers)
        self.task_scheduler = TaskScheduler(self.worker_pool)
        self.load_balancer = LoadBalancer()
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_task_counts: Dict[str, int] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}  # session_id -> request timestamps
        self.max_requests_per_minute = 60
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the concurrent processor"""
        await self.task_scheduler.start()
        
        # Register default agents (ports 8306-8315)
        for port in range(8306, 8316):
            self.load_balancer.register_agent(str(port), capacity=5)
        
        self.logger.info("Concurrent processor started")
    
    async def stop(self):
        """Stop the concurrent processor"""
        await self.task_scheduler.stop()
        self.logger.info("Concurrent processor stopped")
    
    async def submit_agent_task(self,
                               agent_id: str,
                               function: Callable,
                               args: tuple = (),
                               kwargs: dict = None,
                               session_id: Optional[str] = None,
                               priority: TaskPriority = TaskPriority.NORMAL,
                               timeout_seconds: int = 300) -> Optional[str]:
        """Submit a task for a specific agent"""
        
        # Check session limits
        if session_id and not self._check_session_limits(session_id):
            raise Exception("Session limit exceeded")
        
        # Check rate limits
        if session_id and not self._check_rate_limit(session_id):
            raise Exception("Rate limit exceeded")
        
        # Check if agent can handle the task
        if not self.load_balancer.assign_task(agent_id):
            # Try to find alternative agent
            alternative_agent = self.load_balancer.select_agent()
            if alternative_agent:
                agent_id = alternative_agent
                if not self.load_balancer.assign_task(agent_id):
                    raise Exception("No available agents")
            else:
                raise Exception("No available agents")
        
        # Submit task
        task_id = await self.task_scheduler.submit_task(
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            metadata={'agent_id': agent_id}
        )
        
        # Track session task
        if session_id:
            self.session_task_counts[session_id] = self.session_task_counts.get(session_id, 0) + 1
        
        # Add completion callback to update load balancer
        self.task_scheduler.add_task_callback(
            task_id,
            lambda: self.load_balancer.complete_task(agent_id)
        )
        
        return task_id
    
    def _check_session_limits(self, session_id: str) -> bool:
        """Check if session is within limits"""
        current_tasks = self.session_task_counts.get(session_id, 0)
        return current_tasks < 10  # Max 10 concurrent tasks per session
    
    def _check_rate_limit(self, session_id: str) -> bool:
        """Check rate limiting for session"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        if session_id not in self.rate_limits:
            self.rate_limits[session_id] = []
        
        # Remove old requests
        self.rate_limits[session_id] = [
            timestamp for timestamp in self.rate_limits[session_id]
            if timestamp > cutoff
        ]
        
        # Check limit
        if len(self.rate_limits[session_id]) >= self.max_requests_per_minute:
            return False
        
        # Add current request
        self.rate_limits[session_id].append(now)
        return True
    
    async def get_task_result(self, task_id: str, timeout_seconds: int = 30) -> Any:
        """Wait for task result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            task = await self.task_scheduler.get_task_status(task_id)
            
            if task is None:
                raise Exception(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise task.error or Exception("Task failed")
            elif task.status == TaskStatus.CANCELLED:
                raise Exception("Task was cancelled")
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout_seconds}s")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'concurrent_processor': {
                'max_workers': self.max_workers,
                'max_concurrent_users': self.max_concurrent_users,
                'active_sessions': len(self.active_sessions),
                'total_session_tasks': sum(self.session_task_counts.values())
            },
            'task_scheduler': self.task_scheduler.get_queue_stats(),
            'load_balancer': self.load_balancer.get_load_stats(),
            'rate_limiting': {
                'sessions_with_limits': len(self.rate_limits),
                'max_requests_per_minute': self.max_requests_per_minute
            }
        }
    
    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations"""
        recommendations = []
        stats = self.get_system_stats()
        
        # Check worker utilization
        worker_stats = stats['task_scheduler']['worker_pool_stats']
        if worker_stats['active_workers'] / worker_stats['max_workers'] > 0.8:
            recommendations.append({
                'type': 'high_worker_utilization',
                'priority': 'medium',
                'title': 'High Worker Utilization',
                'description': f"Worker utilization is {worker_stats['active_workers']}/{worker_stats['max_workers']}",
                'actions': ['Increase worker pool size', 'Optimize task execution time']
            })
        
        # Check queue size
        if worker_stats['queue_size'] > 50:
            recommendations.append({
                'type': 'large_queue_size',
                'priority': 'high',
                'title': 'Large Task Queue',
                'description': f"Task queue has {worker_stats['queue_size']} pending tasks",
                'actions': ['Increase worker pool size', 'Implement task prioritization', 'Add more agent instances']
            })
        
        # Check load balancer
        lb_stats = stats['load_balancer']
        if lb_stats['overall_utilization'] > 0.8:
            recommendations.append({
                'type': 'high_agent_utilization',
                'priority': 'high',
                'title': 'High Agent Utilization',
                'description': f"Overall agent utilization is {lb_stats['overall_utilization']:.1%}",
                'actions': ['Add more agent instances', 'Increase agent capacity', 'Implement request queuing']
            })
        
        return recommendations

# Global concurrent processor instance
concurrent_processor = ConcurrentProcessor()