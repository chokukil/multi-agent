#!/usr/bin/env python3
"""
🔍 Enhanced Langfuse + A2A Integration Example
Demonstrates comprehensive multi-agent logging and streaming with CherryAI

This example shows:
1. Session-based tracing across multiple agents
2. Real-time streaming with trace propagation
3. Enhanced distributed tracing
4. Performance monitoring
5. Error tracking and recovery
"""

import asyncio
import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Core CherryAI imports
from core.langfuse_session_tracer import get_session_tracer
from core.langfuse_otel_integration import get_otel_integration, trace_a2a_agent
from core.enhanced_a2a_communicator import get_enhanced_a2a_communicator
from core.llm_factory import create_llm_instance

# A2A imports
from a2a.types import Message, TextPart, TaskState
from a2a.utils import new_agent_text_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMultiAgentExample:
    """
    Example demonstrating enhanced Langfuse integration with A2A multi-agent system
    """
    
    def __init__(self):
        """Initialize the enhanced multi-agent system"""
        # Initialize tracing systems
        self.session_tracer = get_session_tracer()
        self.otel_integration = get_otel_integration()
        self.communicator = get_enhanced_a2a_communicator()
        
        # Initialize LLM with automatic langfuse callbacks
        self.llm = create_llm_instance(
            provider="OPENAI",
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # A2A agent URLs (adjust ports as needed)
        self.agents = {
            "data_loader": "http://localhost:8306",
            "eda_agent": "http://localhost:8307", 
            "ml_agent": "http://localhost:8308",
            "visualizer": "http://localhost:8309"
        }
        
        logger.info("✅ Enhanced multi-agent system initialized")
    
    async def run_comprehensive_example(self):
        """Run a comprehensive example showing all integration features"""
        
        print("🚀 Starting Enhanced Langfuse + A2A Integration Example")
        print("=" * 60)
        
        # Example 1: Session-based multi-agent workflow
        await self.example_1_session_based_workflow()
        
        # Example 2: Real-time streaming with tracing
        await self.example_2_streaming_workflow()
        
        # Example 3: Enhanced distributed tracing
        await self.example_3_distributed_tracing()
        
        # Example 4: Performance monitoring
        await self.example_4_performance_monitoring()
        
        # Example 5: Error tracking and recovery
        await self.example_5_error_tracking()
        
        print("\n✅ All examples completed successfully!")
    
    async def example_1_session_based_workflow(self):
        """Example 1: Session-based multi-agent workflow"""
        print("\n📊 Example 1: Session-Based Multi-Agent Workflow")
        print("-" * 50)
        
        # Start session with user identification
        user_query = "Analyze sales data and create ML model"
        user_id = os.getenv("EMP_NO", "demo_user")
        
        session_id = self.session_tracer.start_user_session(
            user_query=user_query,
            user_id=user_id,
            session_metadata={
                "interface": "example_script",
                "environment": "development",
                "example_type": "session_based_workflow",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        print(f"📝 Started session: {session_id}")
        
        try:
            # Agent 1: Data Loading
            with self.session_tracer.trace_agent_execution(
                "DataLoader", 
                "Load sales data from CSV"
            ) as span1:
                print("🔄 Agent 1: Loading data...")
                data_result = await self._simulate_data_loading()
                print(f"✅ Data loaded: {data_result['rows']} rows")
            
            # Agent 2: EDA
            with self.session_tracer.trace_agent_execution(
                "EDA Agent", 
                "Perform exploratory data analysis"
            ) as span2:
                print("🔄 Agent 2: Performing EDA...")
                eda_result = await self._simulate_eda_analysis(data_result)
                print(f"✅ EDA completed: {eda_result['insights']} insights found")
            
            # Agent 3: ML Model
            with self.session_tracer.trace_agent_execution(
                "ML Agent", 
                "Build predictive model"
            ) as span3:
                print("🔄 Agent 3: Building ML model...")
                ml_result = await self._simulate_ml_modeling(data_result, eda_result)
                print(f"✅ Model trained: {ml_result['accuracy']:.3f} accuracy")
            
            # End session with comprehensive summary
            final_result = {
                "data_analysis": {
                    "rows_processed": data_result['rows'],
                    "insights_found": eda_result['insights'],
                    "model_accuracy": ml_result['accuracy']
                },
                "workflow_completed": True,
                "total_agents": 3
            }
            
            session_summary = {
                "agents_used": ["DataLoader", "EDA Agent", "ML Agent"],
                "success": True,
                "processing_time": time.time() - session_id,
                "user_satisfaction": "high"
            }
            
            self.session_tracer.end_user_session(final_result, session_summary)
            print(f"📋 Session ended with summary: {len(session_summary)} metrics")
            
        except Exception as e:
            print(f"❌ Session failed: {e}")
            self.session_tracer.end_user_session(
                {"error": str(e)}, 
                {"success": False, "error_type": type(e).__name__}
            )
    
    async def example_2_streaming_workflow(self):
        """Example 2: Real-time streaming with tracing"""
        print("\n🌊 Example 2: Real-Time Streaming Workflow")
        print("-" * 50)
        
        # Start streaming session
        query = "Generate real-time data analysis report"
        session_id = self.session_tracer.start_user_session(
            query, 
            os.getenv("EMP_NO", "demo_user"),
            {"streaming": True, "example_type": "streaming_workflow"}
        )
        
        print(f"📝 Started streaming session: {session_id}")
        
        # Streaming callback with tracing
        chunks_received = 0
        
        async def stream_callback(content: str):
            nonlocal chunks_received
            chunks_received += 1
            print(f"📡 Streaming chunk {chunks_received}: {content[:50]}...")
            
            # Additional streaming analytics
            if self.session_tracer:
                self.session_tracer.trace_agent_internal_logic(
                    agent_name="StreamingHandler",
                    operation="process_chunk",
                    input_data={"chunk_index": chunks_received},
                    output_data={"content_length": len(content)},
                    operation_metadata={"timestamp": time.time()}
                )
        
        # Execute streaming workflow
        with self.session_tracer.trace_agent_execution(
            "StreamingOrchestrator", 
            "Real-time streaming analysis"
        ) as span:
            
            # Simulate streaming response
            stream_result = await self._simulate_streaming_response(stream_callback)
            
            print(f"✅ Streaming completed: {chunks_received} chunks received")
        
        # End streaming session
        self.session_tracer.end_user_session(
            {"chunks_processed": chunks_received, "streaming_success": True},
            {"streaming_mode": True, "chunk_count": chunks_received}
        )
    
    async def example_3_distributed_tracing(self):
        """Example 3: Enhanced distributed tracing with OpenTelemetry"""
        print("\n🔗 Example 3: Enhanced Distributed Tracing")
        print("-" * 50)
        
        if not self.otel_integration.enabled:
            print("⚠️ OpenTelemetry integration not available - skipping example")
            return
        
        # Distributed tracing across multiple agents
        with self.otel_integration.trace_a2a_agent_execution(
            "DistributedOrchestrator",
            "Multi-agent distributed workflow",
            {"workflow_type": "distributed", "agent_count": len(self.agents)}
        ) as main_span:
            
            print("🔄 Starting distributed workflow...")
            
            # Agent health check with tracing
            health_results = await self.communicator.discover_agents(
                list(self.agents.values())
            )
            
            healthy_agents = [name for name, result in health_results.items() 
                            if result.get("status") == "healthy"]
            
            print(f"✅ Health check: {len(healthy_agents)} healthy agents")
            
            # Distributed execution with trace context propagation
            tasks = []
            for agent_name, agent_url in self.agents.items():
                if f"Agent-{agent_url.split(':')[2]}" in healthy_agents:
                    task = self._execute_agent_with_distributed_tracing(
                        agent_name, agent_url, main_span
                    )
                    tasks.append(task)
            
            # Execute all agents in parallel with distributed tracing
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_executions = len([r for r in results if not isinstance(r, Exception)])
            print(f"✅ Distributed execution: {successful_executions}/{len(tasks)} successful")
            
            # Add distributed tracing metrics
            if main_span:
                main_span.set_attribute("distributed.agents_total", len(self.agents))
                main_span.set_attribute("distributed.agents_healthy", len(healthy_agents))
                main_span.set_attribute("distributed.executions_successful", successful_executions)
    
    async def example_4_performance_monitoring(self):
        """Example 4: Performance monitoring and metrics"""
        print("\n📈 Example 4: Performance Monitoring")
        print("-" * 50)
        
        # Start performance monitoring session
        session_id = self.session_tracer.start_user_session(
            "Performance monitoring test",
            os.getenv("EMP_NO", "demo_user"),
            {"monitoring": True, "example_type": "performance_monitoring"}
        )
        
        # Monitor different performance aspects
        performance_metrics = {}
        
        # CPU-intensive task
        with self.session_tracer.trace_agent_execution(
            "CPUAgent", 
            "CPU-intensive calculation"
        ) as span:
            start_time = time.time()
            await self._simulate_cpu_intensive_task()
            cpu_time = time.time() - start_time
            performance_metrics["cpu_task_time"] = cpu_time
            print(f"⚡ CPU task completed in {cpu_time:.3f}s")
        
        # Memory-intensive task
        with self.session_tracer.trace_agent_execution(
            "MemoryAgent", 
            "Memory-intensive operation"
        ) as span:
            start_time = time.time()
            await self._simulate_memory_intensive_task()
            memory_time = time.time() - start_time
            performance_metrics["memory_task_time"] = memory_time
            print(f"🧠 Memory task completed in {memory_time:.3f}s")
        
        # Network-intensive task
        with self.session_tracer.trace_agent_execution(
            "NetworkAgent", 
            "Network-intensive communication"
        ) as span:
            start_time = time.time()
            await self._simulate_network_intensive_task()
            network_time = time.time() - start_time
            performance_metrics["network_task_time"] = network_time
            print(f"🌐 Network task completed in {network_time:.3f}s")
        
        # Record custom performance metrics
        total_time = sum(performance_metrics.values())
        self.session_tracer.record_agent_result(
            agent_name="PerformanceMonitor",
            result=performance_metrics,
            confidence=0.95,
            artifacts=[{
                "name": "performance_report",
                "type": "metrics",
                "data": performance_metrics
            }]
        )
        
        print(f"📊 Total performance monitoring time: {total_time:.3f}s")
        
        # End performance monitoring session
        self.session_tracer.end_user_session(
            {"performance_metrics": performance_metrics, "total_time": total_time},
            {"monitoring_completed": True, "metrics_count": len(performance_metrics)}
        )
    
    async def example_5_error_tracking(self):
        """Example 5: Error tracking and recovery"""
        print("\n🚨 Example 5: Error Tracking and Recovery")
        print("-" * 50)
        
        # Start error tracking session
        session_id = self.session_tracer.start_user_session(
            "Error tracking and recovery test",
            os.getenv("EMP_NO", "demo_user"),
            {"error_testing": True, "example_type": "error_tracking"}
        )
        
        # Simulate various error scenarios
        error_scenarios = [
            ("NetworkError", self._simulate_network_error),
            ("DataError", self._simulate_data_error),
            ("ProcessingError", self._simulate_processing_error)
        ]
        
        recovery_attempts = 0
        successful_recoveries = 0
        
        for error_type, error_func in error_scenarios:
            print(f"🔄 Testing {error_type}...")
            
            with self.session_tracer.trace_agent_execution(
                f"ErrorTestAgent_{error_type}",
                f"Testing {error_type} scenario"
            ) as span:
                
                try:
                    await error_func()
                    print(f"✅ {error_type}: No error occurred")
                    
                except Exception as e:
                    recovery_attempts += 1
                    print(f"❌ {error_type}: {e}")
                    
                    # Track error in session
                    self.session_tracer.trace_agent_internal_logic(
                        agent_name=f"ErrorHandler_{error_type}",
                        operation="error_occurred",
                        input_data={"error_type": error_type},
                        output_data={"error_message": str(e)},
                        operation_metadata={
                            "error_class": type(e).__name__,
                            "recovery_attempted": True,
                            "timestamp": time.time()
                        }
                    )
                    
                    # Attempt recovery
                    try:
                        await self._simulate_error_recovery(error_type)
                        successful_recoveries += 1
                        print(f"✅ {error_type}: Recovery successful")
                    except Exception as recovery_error:
                        print(f"❌ {error_type}: Recovery failed - {recovery_error}")
        
        # Summary of error tracking
        error_summary = {
            "scenarios_tested": len(error_scenarios),
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0
        }
        
        print(f"📋 Error tracking summary: {successful_recoveries}/{recovery_attempts} recoveries successful")
        
        # End error tracking session
        self.session_tracer.end_user_session(
            {"error_tracking": error_summary, "testing_completed": True},
            {"error_scenarios": len(error_scenarios), "recovery_success": successful_recoveries}
        )
    
    # Helper methods for simulation
    
    async def _simulate_data_loading(self) -> Dict[str, Any]:
        """Simulate data loading operation"""
        await asyncio.sleep(0.5)  # Simulate processing time
        return {
            "rows": 10000,
            "columns": 15,
            "data_quality": "good",
            "processing_time": 0.5
        }
    
    async def _simulate_eda_analysis(self, data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate EDA analysis"""
        await asyncio.sleep(0.8)  # Simulate analysis time
        return {
            "insights": 8,
            "correlations": 12,
            "outliers": 45,
            "data_quality_score": 0.85
        }
    
    async def _simulate_ml_modeling(self, data_result: Dict[str, Any], eda_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate ML modeling"""
        await asyncio.sleep(1.2)  # Simulate training time
        return {
            "accuracy": 0.847,
            "precision": 0.823,
            "recall": 0.879,
            "f1_score": 0.850
        }
    
    async def _simulate_streaming_response(self, callback) -> Dict[str, Any]:
        """Simulate streaming response"""
        chunks = [
            "📊 Starting data analysis...",
            "🔍 Loading dataset (10,000 rows)...",
            "📈 Performing statistical analysis...",
            "🤖 Training machine learning model...",
            "✅ Analysis complete! Model accuracy: 84.7%"
        ]
        
        for i, chunk in enumerate(chunks):
            await callback(chunk)
            await asyncio.sleep(0.3)  # Simulate processing delay
        
        return {"chunks_sent": len(chunks), "streaming_successful": True}
    
    @trace_a2a_agent(task_description="Distributed agent execution")
    async def _execute_agent_with_distributed_tracing(self, agent_name: str, agent_url: str, parent_span):
        """Execute agent with distributed tracing"""
        await asyncio.sleep(0.2)  # Simulate network delay
        
        # Simulate agent execution
        return {
            "agent": agent_name,
            "status": "success",
            "execution_time": 0.2,
            "result": f"Agent {agent_name} executed successfully"
        }
    
    async def _simulate_cpu_intensive_task(self):
        """Simulate CPU-intensive task"""
        # Simulate CPU work
        total = 0
        for i in range(100000):
            total += i * i
        await asyncio.sleep(0.1)
        return total
    
    async def _simulate_memory_intensive_task(self):
        """Simulate memory-intensive task"""
        # Simulate memory allocation
        data = [i for i in range(10000)]
        await asyncio.sleep(0.1)
        return len(data)
    
    async def _simulate_network_intensive_task(self):
        """Simulate network-intensive task"""
        # Simulate network operations
        await asyncio.sleep(0.3)
        return "Network operation completed"
    
    async def _simulate_network_error(self):
        """Simulate network error"""
        await asyncio.sleep(0.1)
        raise ConnectionError("Simulated network connection failed")
    
    async def _simulate_data_error(self):
        """Simulate data error"""
        await asyncio.sleep(0.1)
        raise ValueError("Simulated data validation error")
    
    async def _simulate_processing_error(self):
        """Simulate processing error"""
        await asyncio.sleep(0.1)
        raise RuntimeError("Simulated processing error")
    
    async def _simulate_error_recovery(self, error_type: str):
        """Simulate error recovery"""
        await asyncio.sleep(0.2)
        if error_type == "ProcessingError":
            # Simulate recovery failure for demonstration
            raise RuntimeError("Recovery failed for processing error")
        # Other errors recover successfully
        return f"Recovery successful for {error_type}"


async def main():
    """Main execution function"""
    print("🔍 Enhanced Langfuse + A2A Integration Example")
    print("=" * 60)
    
    # Initialize example system
    example = EnhancedMultiAgentExample()
    
    # Run comprehensive example
    await example.run_comprehensive_example()
    
    print("\n🎉 Example execution completed!")
    print("📊 Check your Langfuse dashboard for comprehensive tracing data")
    print("🔗 All session data, agent interactions, and performance metrics are captured")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 