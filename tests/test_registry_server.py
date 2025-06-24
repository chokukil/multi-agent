import pytest
from httpx import AsyncClient

from a2a.client import A2AClient
from a2a.types import AgentCard, AgentSkill, MessageSendParams

from tests.conftest import registry_server


@pytest.mark.asyncio
async def test_register_and_list_agents(registry_server):
    """
    Test that a new agent can register and that it appears in the agent list.
    """
    async with AsyncClient(base_url=registry_server, app=registry_server.app) as ac:
        client = A2AClient(httpx_client=ac)

        # 1. Register a new agent
        agent_to_register = AgentCard(
            name="test-agent-123",
            version="0.1.0",
            description="A test agent",
            url=f"{registry_server.url}/test-agent",
            skills=[],
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            capabilities={"streaming": False},
        )

        register_params = MessageSendParams(
            message={
                "role": "user",
                "parts": [
                    {
                        "kind": "data",
                        "data": {"agent_card": agent_to_register.model_dump(mode="json")},
                    }
                ],
            }
        )
        
        register_task = await client.message_send("register_agent", register_params)
        assert register_task.status.state == "completed"
        registered_card = AgentCard.model_validate(register_task.status.result)
        assert registered_card.name == "test-agent-123"

        # 2. List agents and verify the new agent is there
        list_params = MessageSendParams(message={"role": "user", "parts": []})
        list_task = await client.message_send("list_agents", list_params)
        
        assert list_task.status.state == "completed"
        agent_list = list_task.status.result
        assert isinstance(agent_list, list)
        assert len(agent_list) == 1
        assert agent_list[0] == "test-agent-123"
