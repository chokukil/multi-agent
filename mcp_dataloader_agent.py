import pandas as pd
import os
import uvicorn
import logging
from typing import Dict, Any

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.types import AgentCard, AgentSkill, Message, Task
from a2a.utils.message import new_agent_text_message, get_message_text
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

from core.data_manager import DataManager

# 1. Define Agent Skills (functions)
def load_data(file_path: str, output_df_id: str) -> Message:
    # ... (function implementation is fine) ...

# 2. Implement the AgentExecutor
class SkillBasedAgentExecutor(AgentExecutor):
    def __init__(self, skill_handlers: Dict[str, Any]):
        self._skill_handlers = skill_handlers

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        skill_id = context.method
        handler = self._skill_handlers.get(skill_id)
        
        if not handler:
            error_message = new_agent_text_message(f"Skill '{skill_id}' not found.")
            await event_queue.put(error_message)
            return

        try:
            params = context.params or {}
            result = handler(**params)
            await event_queue.put(result)
        except Exception as e:
            error_message = new_agent_text_message(f"Error executing skill '{skill_id}': {e}")
            await event_queue.put(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Not implemented for this simple agent
        pass

# 3. Wire everything together
skill_handlers: Dict[str, Any] = {
    "load_data": load_data,
}

SERVER_HOST = os.getenv("DATALOADER_AGENT_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("DATALOADER_AGENT_PORT", 8001))

agent_card = AgentCard(
    # ... (agent_card definition is mostly fine, but requires some mandatory fields) ...
    name="mcp_dataloader_agent",
    description="An agent responsible for loading data from files.",
    version="0.1.0",
    url=f"http://{SERVER_HOST}:{SERVER_PORT}",
    capabilities={"streaming": False},
    defaultInputModes=["application/json"],
    defaultOutputModes=["application/json"],
    skills=[
        AgentSkill(
            id="load_data",
            name="load_data_skill",
            description="Loads a data file (CSV or Excel) and stores it in the data manager.",
            tags=["data", "loader"],
        ),
    ]
)

# Setup Server components
agent_executor = SkillBasedAgentExecutor(skill_handlers=skill_handlers)
task_store = InMemoryTaskStore()
handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=task_store)
a2a_app = A2AFastAPIApplication(agent_card=agent_card, http_handler=handler)
app = a2a_app.build()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"🚀 Starting MCP DataLoaderAgent A2A Server at http://{SERVER_HOST}:{SERVER_PORT}...")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT) 