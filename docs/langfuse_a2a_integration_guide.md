# 🔍 CherryAI Langfuse + A2A SDK Integration Guide

## Overview

CherryAI features a comprehensive **Langfuse integration with A2A SDK** for multi-agent logging and streaming. This guide covers both the existing implementation and enhancement opportunities.

## 📋 Current Integration Status

### ✅ **Implemented Features**

1. **Session-Based Unified Tracing**
   - All operations from one user query grouped under single session
   - Hierarchical structure: User Query → Agent → Internal Logic → Details
   - EMP_NO integration for user identification

2. **Real-Time Streaming with Tracing**
   - Maintains trace context during streaming
   - Chunk-level tracking and performance monitoring
   - A2A SDK 0.2.9 compliant streaming implementation

3. **Multi-Agent Orchestration**
   - 9 A2A servers (ports 8306-8314) with integrated tracing
   - Complex workflow tracking across agents
   - Automatic LLM callback injection

4. **Deep Library Integration**
   - AI-Data-Science-Team wrapper for internal tracking
   - LLM step-by-step monitoring
   - Artifact generation and tracking

## 🚀 Architecture Components

### **1. Core Tracing System**

```python
# SessionBasedTracer - Central tracing system
from core.langfuse_session_tracer import get_session_tracer

tracer = get_session_tracer()
session_id = tracer.start_user_session(
    user_query="Analyze sales data",
    user_id=os.getenv("EMP_NO"),
    session_metadata={
        "interface": "streamlit",
        "environment": "production"
    }
)
```

### **2. A2A Agent Integration**

```python
# Enhanced A2A Executor with automatic tracing
from core.langfuse_enhanced_a2a_executor import LangfuseEnhancedA2AExecutor

class MyA2AAgent(LangfuseEnhancedA2AExecutor):
    def __init__(self):
        super().__init__("My Custom Agent")
        
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Automatic tracing of all operations
        with self.trace_agent_execution("data_analysis") as span:
            # Your agent logic here
            result = await self.process_data(context.get_user_input())
            return result
```

### **3. Streaming Integration**

```python
# Real-time streaming with trace propagation
from a2a_ds_servers.a2a_orchestrator import RealTimeStreamingTaskUpdater

task_updater = RealTimeStreamingTaskUpdater(
    event_queue, task_id, context_id
)

# Stream with automatic tracing
for chunk in response_stream:
    await task_updater.stream_chunk(chunk, final=False)
    # Langfuse automatically captures streaming data
```

### **4. Enhanced Distribution Tracing (New)**

```python
# OpenTelemetry + Langfuse integration
from core.langfuse_otel_integration import get_otel_integration
from core.enhanced_a2a_communicator import get_enhanced_a2a_communicator

otel = get_otel_integration()
communicator = get_enhanced_a2a_communicator()

# Distributed tracing across agents
with otel.trace_a2a_agent_execution("DataAnalysis", "Process CSV data") as span:
    result = await communicator.send_message_with_streaming(
        agent_url="http://localhost:8306",
        instruction="Analyze the uploaded CSV file",
        stream_callback=my_stream_callback
    )
```

## 🔧 Configuration

### **Environment Variables**

```bash
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk_...
LANGFUSE_SECRET_KEY=sk_...
LANGFUSE_HOST=http://localhost:3000

# User Identification
EMP_NO=12345  # Employee number for user tracking

# Logging Configuration
LOGGING_PROVIDER=langfuse  # or 'both' for langfuse + langsmith

# OpenTelemetry (Optional)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=cherry-ai-a2a-system
```

### **LLM Factory Integration**

```python
# Automatic langfuse callback injection
from core.llm_factory import create_llm_instance

# LLM instances automatically include langfuse callbacks
llm = create_llm_instance(
    provider="OPENAI",
    model="gpt-4o",
    temperature=0.7
)
# Langfuse callback automatically added when LOGGING_PROVIDER=langfuse
```

## 📊 Multi-Agent Workflow Tracking

### **1. Session-Based Grouping**

```python
# All related operations under single session
async def process_user_query(query: str):
    session_tracer = get_session_tracer()
    session_id = session_tracer.start_user_session(query, user_id)
    
    # Agent 1: Data Loading
    with session_tracer.trace_agent_execution("DataLoader", "Load CSV") as span1:
        data = await load_data_agent.execute(query)
    
    # Agent 2: Analysis
    with session_tracer.trace_agent_execution("DataAnalyst", "Analyze data") as span2:
        analysis = await analyst_agent.execute(data)
    
    # Agent 3: Visualization
    with session_tracer.trace_agent_execution("Visualizer", "Create plots") as span3:
        plots = await viz_agent.execute(analysis)
    
    # End session with summary
    session_tracer.end_user_session(
        final_result={"analysis": analysis, "plots": plots},
        session_summary={"agents_used": 3, "success": True}
    )
```

### **2. Streaming Workflow**

```python
async def streaming_multi_agent_workflow(query: str, stream_callback):
    orchestrator = UniversalIntelligentOrchestratorV8()
    
    # Real-time streaming with trace propagation
    async def traced_stream_callback(content: str):
        # Streaming content automatically traced
        await stream_callback(content)
    
    # Execute with comprehensive tracing
    result = await orchestrator.execute_with_streaming(
        query, 
        stream_callback=traced_stream_callback
    )
    
    # All agent interactions, streaming chunks, and results traced
```

## 🔍 Advanced Features

### **1. AI-Data-Science-Team Deep Tracking**

```python
# Automatic tracking of library internals
from core.langfuse_ai_ds_team_wrapper import LangfuseAIDataScienceTeamWrapper

# Wrapper automatically tracks:
# - LLM prompt/response pairs
# - Code generation steps
# - Data transformation operations
# - Execution results and artifacts

wrapper = LangfuseAIDataScienceTeamWrapper(session_tracer, "EDA Agent")
result = wrapper.trace_ai_ds_team_invoke(
    agent, 
    'invoke_agent',
    user_instructions="Perform EDA on sales data",
    data_raw=dataframe
)
```

### **2. Performance Monitoring**

```python
# Automatic performance tracking
class PerformanceTrackedAgent(LangfuseEnhancedA2AExecutor):
    async def execute(self, context, event_queue):
        start_time = time.time()
        
        # Automatic performance metrics
        with self.trace_agent_execution("data_processing") as span:
            result = await self.process_data()
            
            # Performance automatically tracked:
            # - Execution time
            # - Memory usage
            # - Token consumption
            # - Error rates
            
        return result
```

### **3. Error Tracking and Recovery**

```python
# Comprehensive error tracking
async def fault_tolerant_execution(query: str):
    try:
        with session_tracer.trace_agent_execution("MainAgent", query) as span:
            result = await risky_operation()
            return result
    except Exception as e:
        # Error automatically tracked with:
        # - Stack trace
        # - Context information
        # - Recovery attempts
        # - Fallback strategies
        
        session_tracer.record_error(
            error=e,
            context={"query": query, "agent": "MainAgent"},
            recovery_attempted=True
        )
        
        # Attempt recovery
        return await fallback_strategy()
```

## 📈 Monitoring and Analytics

### **1. Langfuse Dashboard Views**

- **Session Overview**: All operations for each user query
- **Agent Performance**: Individual agent execution metrics
- **Streaming Analytics**: Real-time streaming performance
- **Error Analysis**: Comprehensive error tracking
- **User Activity**: EMP_NO-based user behavior analysis

### **2. Custom Metrics**

```python
# Custom metrics integration
session_tracer.record_custom_metric(
    name="data_processing_efficiency",
    value=0.95,
    metadata={
        "rows_processed": 10000,
        "processing_time": 30.5,
        "agent": "DataProcessor"
    }
)
```

### **3. A2A-Specific Analytics**

```python
# A2A protocol-specific tracking
communicator = get_enhanced_a2a_communicator()
agents_health = await communicator.discover_agents([
    "http://localhost:8306",
    "http://localhost:8307", 
    "http://localhost:8308"
])

# Health metrics automatically tracked:
# - Response times
# - Success rates
# - Agent availability
# - Protocol compliance
```

## 🚀 Best Practices

### **1. Session Management**

```python
# Always use session-based tracing for user queries
async def handle_user_request(query: str):
    tracer = get_session_tracer()
    session_id = tracer.start_user_session(
        query, 
        user_id=os.getenv("EMP_NO"),
        session_metadata={
            "interface": "streamlit",
            "request_type": "analysis"
        }
    )
    
    try:
        # Your processing logic
        result = await process_query(query)
        tracer.end_user_session(result, {"success": True})
    except Exception as e:
        tracer.end_user_session({"error": str(e)}, {"success": False})
```

### **2. Agent Instrumentation**

```python
# Use enhanced executors for comprehensive tracking
class MyAgent(LangfuseEnhancedA2AExecutor):
    def __init__(self):
        super().__init__("My Agent Name")
        
    async def execute(self, context, event_queue):
        # Create AI-DS wrapper for deep tracking
        wrapper = self.create_ai_ds_wrapper("my_operation")
        
        # Execute with tracking
        result = self.trace_ai_ds_team_invoke(
            self.agent, 
            'invoke_agent',
            user_instructions=context.get_user_input()
        )
        
        return result
```

### **3. Streaming Best Practices**

```python
# Maintain trace context during streaming
async def streaming_execution(query: str, stream_callback):
    task_updater = RealTimeStreamingTaskUpdater(
        event_queue, task_id, context_id
    )
    
    # Stream with automatic tracing
    response = await llm.astream(query)
    
    full_response = ""
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            await task_updater.stream_chunk(content)
    
    # Final response with complete trace
    await task_updater.stream_final_response(full_response)
```

## 📚 Integration Examples

### **Example 1: Multi-Agent Data Analysis**

```python
async def analyze_sales_data(csv_file: str):
    # Start session
    tracer = get_session_tracer()
    session_id = tracer.start_user_session(
        f"Analyze sales data: {csv_file}",
        user_id=os.getenv("EMP_NO")
    )
    
    # Agent 1: Data Loading
    with tracer.trace_agent_execution("DataLoader", "Load CSV") as span:
        data = await data_loader.load_csv(csv_file)
    
    # Agent 2: EDA
    with tracer.trace_agent_execution("EDA", "Exploratory Analysis") as span:
        eda_result = await eda_agent.analyze(data)
    
    # Agent 3: Modeling
    with tracer.trace_agent_execution("Modeler", "Build ML Model") as span:
        model_result = await ml_agent.build_model(data)
    
    # End session
    tracer.end_user_session({
        "eda": eda_result,
        "model": model_result
    })
```

### **Example 2: Real-Time Streaming Analysis**

```python
async def streaming_analysis(query: str, stream_callback):
    communicator = get_enhanced_a2a_communicator()
    
    # Stream with distributed tracing
    result = await communicator.send_message_with_streaming(
        agent_url="http://localhost:8306",
        instruction=query,
        stream_callback=stream_callback,
        context_data={"analysis_type": "real_time"}
    )
    
    return result
```

## 🔧 Troubleshooting

### **Common Issues**

1. **Session Not Starting**
   ```python
   # Check Langfuse configuration
   tracer = get_session_tracer()
   if not tracer.enabled:
       print("Langfuse not configured properly")
   ```

2. **Missing Trace Data**
   ```python
   # Ensure EMP_NO is set
   if not os.getenv("EMP_NO"):
       print("EMP_NO not set - user identification disabled")
   ```

3. **Streaming Issues**
   ```python
   # Check A2A agent health
   communicator = get_enhanced_a2a_communicator()
   health = await communicator.health_check("http://localhost:8306")
   print(f"Agent health: {health}")
   ```

### **Performance Optimization**

```python
# Optimize for high-throughput scenarios
tracer = get_session_tracer()
tracer.configure_batch_processing(
    batch_size=100,
    flush_interval=5.0,
    max_queue_size=1000
)
```

## 📊 Monitoring Dashboard

### **Key Metrics to Track**

1. **Session Metrics**
   - Total sessions per user
   - Session duration
   - Success rate

2. **Agent Performance**
   - Average execution time
   - Error rates
   - Resource utilization

3. **Streaming Analytics**
   - Chunk processing time
   - Streaming latency
   - User engagement

4. **A2A Protocol Health**
   - Agent availability
   - Response times
   - Protocol compliance

## 🎯 Next Steps

1. **Enhanced Analytics**: Custom dashboards for specific use cases
2. **Alert System**: Automated alerts for performance degradation
3. **Distributed Tracing**: Full OpenTelemetry integration
4. **Advanced Metrics**: Custom business metrics integration

---

**Note**: This integration provides comprehensive observability for multi-agent systems with A2A protocol. The existing implementation is production-ready and the enhancement suggestions provide additional capabilities for specific use cases. 