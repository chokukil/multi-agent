#!/usr/bin/env python3
"""A2A HelloWorld Agent."""

import logging
import uvicorn
import click

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

# Local imports
from official_helloworld_executor import HelloWorldAgentExecutor


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
        name="HelloWorld",
        description="A Hello World agent for the A2A protocol.",
        url=f"http://{host}:{port}/",
        version="0.1.0",
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

    logging.info("Starting A2A Helloworld Agent...")
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