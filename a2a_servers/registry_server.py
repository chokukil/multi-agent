import logging
import os
from typing import cast, Callable, AsyncGenerator
from uuid import uuid4

from fastapi import Depends
from langgraph.graph.graph import CompiledGraph

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.agent_execution.simple_request_context_builder import (
    SimpleRequestContextBuilder,
)
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.events.event_queue import Event, EventQueue
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import (
    AgentCard,
    AgentSkill,
    Message,
    MessageSendParams,
    Task,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
)

# --- Agent Definition ---

AGENT_CARD = AgentCard(
    name="registry",
    version="0.1.0",
    description="A registry of all agents in the system.",
    url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
    skills=[
        AgentSkill(
            id="list_agents",
            name="List Agents",
            description="List all available agents.",
            examples=["list agents"],
            tags=["registry"],
        ),
        AgentSkill(
            id="get_agent",
            name="Get Agent",
            description="Get a single agent by name.",
            examples=["get agent 'my-agent'"],
            tags=["registry"],
        ),
    ],
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    capabilities={"streaming": False},
)


# --- Agent Logic ---

# In a real application, this would be a proper registry implementation
AGENT_REGISTRY = {}


async def list_agents(**kwargs) -> list[str]:
    """List all available agents."""
    return list(AGENT_REGISTRY.keys())


async def get_agent(name: str, **kwargs) -> AgentCard | None:
    """Get a single agent by name."""
    return AGENT_REGISTRY.get(name)


async def register_agent(agent_card: dict, **kwargs) -> AgentCard:
    """Register an agent."""
    card = AgentCard.model_validate(agent_card)
    AGENT_REGISTRY[card.name] = card
    return card


# --- Custom Implementations ---
class RegistryAgentExecutor(AgentExecutor):
    def __init__(self, tools: dict[str, Callable]):
        self.tools = tools

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        method_name = context.request.method
        tool = self.tools.get(method_name)

        if not tool:
            # Simplified error handling
            # In a real app, you'd publish a "failed" task status event
            print(f"Tool not found: {method_name}")
            return

        result = await tool(**context.request.params)
        
        task = Task(
            id=context.task_id,
            contextId=context.context_id,
            status=TaskStatus(state=TaskState.completed, result=result),
        )
        await event_queue.put(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


class RegistryRequestHandler(RequestHandler):
    def __init__(self, agent_executor: RegistryAgentExecutor):
        self._agent_executor = agent_executor
        self._queue_manager = InMemoryQueueManager()

    async def on_message_send(self, params: MessageSendParams, context=None) -> Task | Message:
        task_id = str(uuid4())
        context_id = str(uuid4())
        event_queue = await self._queue_manager.create_or_tap(task_id)
        
        request_context = RequestContext(
            request=params,
            task_id=task_id,
            context_id=context_id
        )

        await self._agent_executor.execute(request_context, event_queue)

        # For this simple, non-streaming handler, we expect one event: the final Task
        final_event = await event_queue.get()
        if isinstance(final_event, Task):
            return final_event
        raise Exception("Unexpected event type from agent executor")

    async def on_get_task(self, params: TaskQueryParams, context=None) -> Task | None:
        raise NotImplementedError()

    async def on_cancel_task(self, params: TaskIdParams, context=None) -> Task | None:
        raise NotImplementedError()

    async def on_message_send_stream(self, params: MessageSendParams, context=None) -> AsyncGenerator[Event, None]:
        raise NotImplementedError()
        yield

    async def on_set_task_push_notification_config(self, params: TaskPushNotificationConfig, context=None) -> TaskPushNotificationConfig:
        raise NotImplementedError()

    async def on_get_task_push_notification_config(self, params: TaskIdParams, context=None) -> TaskPushNotificationConfig:
        raise NotImplementedError()
    
    async def on_resubscribe_to_task(self, params: TaskIdParams, context=None) -> AsyncGenerator[Event, None]:
        raise NotImplementedError()
        yield


# --- Server Setup ---

agent_executor = RegistryAgentExecutor(
    tools={
        "list_agents": list_agents,
        "get_agent": get_agent,
        "register_agent": register_agent,
    }
)

request_handler = RegistryRequestHandler(agent_executor=agent_executor)

jsonrpc_handler = JSONRPCHandler(
    agent_card=AGENT_CARD,
    request_handler=request_handler,
)

a2a_app = A2AFastAPIApplication(
    agent_card=AGENT_CARD,
    http_handler=jsonrpc_handler,
    context_builder=SimpleRequestContextBuilder(),
)

app = a2a_app.build()
logging.basicConfig(level=logging.INFO)
