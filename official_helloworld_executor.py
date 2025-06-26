"""A2A HelloWorld Agent Executor."""

from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils.message import new_agent_text_message


class HelloWorldAgent:
    """A very simple agent that returns a fixed message."""

    async def invoke(self) -> str:
        """Runs the agent and returns the final result."""
        return "Hello World"


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