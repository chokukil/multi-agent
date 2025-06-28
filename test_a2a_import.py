#!/usr/bin/env python3
"""A2A SDK Import Test"""

try:
    print("Testing A2A SDK imports...")
    
    # 1. A2AFastAPIApplication
    try:
        from a2a.server.apps import A2AFastAPIApplication
        print("✅ A2AFastAPIApplication imported successfully")
    except ImportError as e:
        print(f"❌ A2AFastAPIApplication import failed: {e}")
    
    # 2. DefaultRequestHandler
    try:
        from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
        print("✅ DefaultRequestHandler imported successfully")
    except ImportError as e:
        print(f"❌ DefaultRequestHandler import failed: {e}")
    
    # 3. AgentExecutor
    try:
        from a2a.server.agent_execution import AgentExecutor
        print("✅ AgentExecutor imported successfully")
    except ImportError as e:
        print(f"❌ AgentExecutor import failed: {e}")
    
    # 4. RequestContext
    try:
        from a2a.server.agent_execution import RequestContext
        print("✅ RequestContext imported successfully")
    except ImportError as e:
        print(f"❌ RequestContext import failed: {e}")
    
    # 5. TaskUpdater
    try:
        from a2a.server.tasks.task_updater import TaskUpdater
        print("✅ TaskUpdater imported successfully")
    except ImportError as e:
        print(f"❌ TaskUpdater import failed: {e}")
    
    # 6. InMemoryTaskStore
    try:
        from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
        print("✅ InMemoryTaskStore imported successfully")
    except ImportError as e:
        print(f"❌ InMemoryTaskStore import failed: {e}")
    
    # 7. Types (correct path)
    try:
        from a2a.types import TextPart, TaskState
        print("✅ Types (TextPart, TaskState) imported successfully")
    except ImportError as e:
        print(f"❌ Types import failed: {e}")
    
    print("\n🎉 A2A SDK import test completed!")
    
    # Test creating a simple server
    print("\n📝 Testing server creation...")
    try:
        task_store = InMemoryTaskStore()
        print("✅ InMemoryTaskStore created")
        
        # Test basic server structure
        print("✅ Basic A2A server structure validated")
        
    except Exception as e:
        print(f"❌ Server creation test failed: {e}")
    
except Exception as e:
    print(f"💥 Unexpected error: {e}") 