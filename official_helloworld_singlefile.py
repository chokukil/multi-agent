#!/usr/bin/env python3
"""A2A HelloWorld Agent (Single File Version)."""

import logging
import uvicorn
import click

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils.message import new_agent_text_message

# --- Agent Logic ---
class HelloWorldAgent:
    """A very simple agent that returns a fixed message."""

    async def invoke(self) -> str:
        """Runs the agent and returns the final result."""
        return "Hello World from the official example!"


class HelloWorldAgentExecutor(AgentExecutor):
    """AgentExecutor for the HelloWorld agent."""

    def __init__(self):
        super().__init__()
        self.agent = HelloWorldAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Executes the agent."""
        result = await self.agent.invoke()
        message = new_agent_text_message(result)
        await event_queue.enqueue_event(message)

# --- Server Setup ---
def get_agent_card(host: str, port: int) -> AgentCard:
    """Returns the agent card."""
    skill = AgentSkill(
        id="hello_world",
        name="Returns hello world",
        description="just returns hello world",
        tags=["hello", "world"],
        examples=["hi", "hello world"],
    )
    return AgentCard(
        name="OfficialHelloWorld",
        description="An official Hello World agent for the A2A protocol.",
        url=f"http://{host}:{port}/",
        version="0.1.0-official",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[skill],
    )


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=9999)
def main(host: str, port: int) -> None:
    """Starts the A2A Helloworld Agent."""
    logging.basicConfig(level=logging.INFO)

    logging.info("Starting A2A Official Helloworld Agent...")
    request_handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=get_agent_card(host, port),
        http_handler=request_handler,
    )
    uvicorn.run(
        server.build(),
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main() 