#!/usr/bin/env python3
"""Simple A2A Server Test"""

try:
    from a2a.server.apps import A2AFastAPIApplication
    from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
    from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
    from a2a.server.tasks.task_updater import TaskUpdater
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.types import TextPart, TaskState, AgentCard
    
    print("✅ All imports successful!")
    
    class SimpleExecutor(AgentExecutor):
        async def execute(self, context: RequestContext) -> None:
            task_store = InMemoryTaskStore()
            event_queue = task_store.get_event_queue()
            task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            
            task_updater.submit()
            task_updater.start_work()
            
            response_text = "Hello from AI_DS_Team!"
            task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=response_text)])
            )
        
        async def cancel(self, context: RequestContext) -> None:
            pass
    
    def create_server():
        task_store = InMemoryTaskStore()
        executor = SimpleExecutor()
        http_handler = DefaultRequestHandler(task_store, executor)
        
        agent_card = AgentCard(
            name="Simple Test Agent",
            version="1.0.0",
            description="A simple test agent",
            author="CherryAI",
            a2a_version="0.2.9"
        )
        
        app = A2AFastAPIApplication(
            agent_card=agent_card,
            http_handler=http_handler
        )
        
        return app
    
    if __name__ == "__main__":
        print("🚀 Starting simple A2A server on port 8300...")
        app = create_server()
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8300)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 