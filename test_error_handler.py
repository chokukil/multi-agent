"""
Test script for LLM Error Handler system.
Demonstrates error handling, recovery, and circuit breaker functionality.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from utils.llm_error_handler import LLMErrorHandler, ErrorContext, ErrorSeverity
from utils.error_recovery_utils import error_logger, error_metrics, error_translator

async def test_error_handler():
    """Test the LLM Error Handler system"""
    print("🧪 Testing LLM Error Handler System")
    print("=" * 50)
    
    # Initialize error handler
    error_handler = LLMErrorHandler()
    
    # Test 1: Basic error handling
    print("\n1. Testing basic error handling...")
    error_context = ErrorContext(
        error_type="ConnectionError",
        error_message="Failed to connect to data_cleaning agent",
        agent_id="data_cleaning",
        user_context={"file_type": "csv", "file_size": "1MB"},
        timestamp=datetime.now()
    )
    
    result = await error_handler.handle_error(error_context)
    print(f"   Result: {result['action']}")
    print(f"   Message: {result['message']}")
    
    # Test 2: Circuit breaker functionality
    print("\n2. Testing circuit breaker...")
    for i in range(6):  # Trigger circuit breaker (threshold = 5)
        error_context.retry_count = i
        result = await error_handler.handle_error(error_context)
        print(f"   Attempt {i+1}: {result['action']}")
        
        if result['action'] == 'circuit_open':
            print("   ⚡ Circuit breaker activated!")
            break
    
    # Test 3: Agent health status
    print("\n3. Checking agent health status...")
    health_status = error_handler.get_agent_health_status()
    for agent_id, status in health_status.items():
        print(f"   {agent_id}: {status['status']} (failures: {status['failure_count']})")
    
    # Test 4: Error translation
    print("\n4. Testing user-friendly error translation...")
    translation = error_translator.translate_error(
        "ConnectionError",
        "Failed to connect to data_cleaning agent",
        {"agent_id": "data_cleaning", "file_type": "csv"}
    )
    print(f"   User message: {translation['user_message']}")
    print(f"   Suggestions: {translation['suggestions']}")
    
    # Test 5: Error metrics
    print("\n5. Testing error metrics...")
    error_metrics.record_error("data_cleaning", "ConnectionError", recovery_attempted=True)
    error_metrics.record_recovery("data_cleaning", success=False, recovery_time=5.2)
    error_metrics.record_circuit_breaker_activation("data_cleaning")
    
    metrics_summary = error_metrics.get_metrics_summary()
    print(f"   Total errors: {metrics_summary['total_errors']}")
    print(f"   Most problematic agent: {metrics_summary['most_problematic_agent']}")
    
    # Test 6: Fallback functionality
    print("\n6. Testing fallback functionality...")
    error_context.retry_count = 5  # Exceed retry limit
    result = await error_handler.handle_error(error_context)
    print(f"   Fallback result: {result['action']}")
    if 'result' in result:
        print(f"   Fallback message: {result['result']['message']}")
    
    # Test 7: Circuit breaker reset
    print("\n7. Testing circuit breaker reset...")
    reset_success = await error_handler.reset_circuit_breaker("data_cleaning")
    print(f"   Reset successful: {reset_success}")
    
    health_status = error_handler.get_agent_health_status()
    if "data_cleaning" in health_status:
        print(f"   New status: {health_status['data_cleaning']['status']}")
    
    print("\n✅ Error handler testing completed!")

async def test_conflict_resolution():
    """Test conflict resolution functionality"""
    print("\n🔄 Testing Conflict Resolution")
    print("=" * 30)
    
    from utils.llm_error_handler import ConflictResolver
    
    resolver = ConflictResolver()
    
    # Simulate conflicting results from multiple agents
    conflicting_results = [
        {
            "agent_id": "eda_tools",
            "conclusion": "Data shows strong positive correlation",
            "confidence": 0.8,
            "timestamp": 1234567890
        },
        {
            "agent_id": "pandas_hub",
            "conclusion": "Data shows weak negative correlation",
            "confidence": 0.6,
            "timestamp": 1234567891
        },
        {
            "agent_id": "visualization",
            "conclusion": "Data shows no significant correlation",
            "confidence": 0.7,
            "timestamp": 1234567892
        }
    ]
    
    context = {
        "analysis_type": "correlation_analysis",
        "dataset": "sales_data.csv"
    }
    
    resolution = await resolver.resolve_conflicts(conflicting_results, context)
    
    print(f"Resolution method: {resolution.get('resolution_method', 'unknown')}")
    print(f"Confidence: {resolution.get('confidence', 'unknown')}")
    print(f"Explanation: {resolution.get('conflict_explanation', 'No explanation')}")
    
    print("\n✅ Conflict resolution testing completed!")

if __name__ == "__main__":
    print("🚀 Starting Error Handler System Tests")
    
    # Run tests
    asyncio.run(test_error_handler())
    asyncio.run(test_conflict_resolution())
    
    print("\n🎉 All tests completed successfully!")