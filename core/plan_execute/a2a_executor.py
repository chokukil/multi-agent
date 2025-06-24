import asyncio
import logging
import re
from typing import Union, List, Dict, Any

import httpx
from uuid import uuid4

from core.agents.agent_registry import AgentRegistry
from core.data_manager import DataManager
from core.schemas.messages import ToolOutput, Plan

logger = logging.getLogger(__name__)


class A2AExecutor:
    """
    Asynchronously executes a plan by making A2A calls to the appropriate agent skills.
    This version uses httpx directly to make JSON-RPC calls, removing dependency on
    the changing a2a-sdk client.
    """

    def __init__(self, data_manager: DataManager, agent_registry: AgentRegistry):
        self.data_manager = data_manager
        self.agent_registry = agent_registry

    def _resolve_dependencies(self, parameters: Dict[str, Any], step_outputs: Dict[str, ToolOutput]) -> Dict[str, Any]:
        """
        Resolves dependencies in parameters using outputs from previous steps.
        Dependencies are in the format {{step_id.data.key}}.
        """
        resolved_params = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                match = re.match(r"\{\{([\w_]+)\.data\.([\w_]+)\}\}", value)
                if match:
                    step_id, data_key = match.groups()
                    if step_id in step_outputs and hasattr(step_outputs[step_id], 'data'):
                        previous_output = step_outputs[step_id].data
                        if previous_output and data_key in previous_output:
                            resolved_params[key] = previous_output[data_key]
                        else:
                            raise ValueError(f"Could not resolve dependency: Key '{data_key}' not found in output of step '{step_id}'.")
                    else:
                        raise ValueError(f"Could not resolve dependency: Step ID '{step_id}' not found or has no data.")
                else:
                    resolved_params[key] = value
            else:
                resolved_params[key] = value
        return resolved_params

    async def _a2a_call(self, agent_name: str, skill_name: str, params: dict, timeout: int) -> ToolOutput:
        """Performs a single, robust A2A call to a registered agent's skill using httpx."""
        try:
            agent_info = self.agent_registry.get_agent_info(agent_name)
            if not agent_info:
                return ToolOutput(stderr=f"Agent '{agent_name}' not found in registry.", exit_code=1)

            agent_url = agent_info.get("url")
            if not agent_url:
                return ToolOutput(stderr=f"Agent '{agent_name}' has no URL in registry.", exit_code=1)

            # Construct the JSON-RPC payload
            payload = {
                "jsonrpc": "2.0",
                "method": skill_name,
                "params": params,
                "id": str(uuid4()),
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(agent_url, json=payload)
                response.raise_for_status() # Raise an exception for 4xx or 5xx status codes

            response_data = response.json()
            
            # Check for JSON-RPC error object
            if "error" in response_data:
                err_msg = response_data["error"].get('message', 'Unknown JSON-RPC error')
                return ToolOutput(stderr=f"Agent '{agent_name}' returned a JSON-RPC error: {err_msg}", exit_code=1)

            result = response_data.get("result")
            if result is None:
                return ToolOutput(stderr=f"Invalid response from agent '{agent_name}': Missing 'result' field.", exit_code=1)

            # Our convention: skill execution result is the 'result' object itself
            # The agent skills return a dictionary with interpretation, new_dataset_id, etc.
            message = result.get("interpretation", f"Skill '{skill_name}' executed successfully.")

            return ToolOutput(stdout=message, data=result, exit_code=0)

        except httpx.TimeoutException:
            logger.error(f"Timeout calling agent '{agent_name}' skill '{skill_name}'")
            return ToolOutput(stderr=f"Request timed out after {timeout} seconds.", exit_code=1)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling agent '{agent_name}': {e.response.status_code} - {e.response.text}")
            return ToolOutput(stderr=f"HTTP error {e.response.status_code}: {e.response.text}", exit_code=1)
        except Exception as e:
            logger.error(f"Error in A2A call to '{agent_name}': {e}", exc_info=True)
            return ToolOutput(stderr=f"An unexpected error occurred: {e}", exit_code=1)

    async def aexecute(self, plan: Plan, timeout: int = 60) -> List[ToolOutput]:
        """Executes a plan object sequentially and returns the list of outputs."""
        step_outputs: Dict[str, ToolOutput] = {}
        all_outputs: List[ToolOutput] = []

        for step in plan.steps:
            logger.info(f"Executing step '{step.id}': Agent '{step.agent_name}', Skill '{step.skill_name}'")
            try:
                resolved_params = self._resolve_dependencies(step.parameters, step_outputs)
            except ValueError as e:
                logger.error(f"Failed to resolve dependencies for step '{step.id}': {e}")
                output = ToolOutput(stderr=f"Dependency resolution failed: {e}", exit_code=1)
                all_outputs.append(output)
                # Halt execution on dependency resolution failure
                return all_outputs

            output = await self._a2a_call(
                agent_name=step.agent_name,
                skill_name=step.skill_name,
                params=resolved_params,
                timeout=timeout,
            )

            step_outputs[step.id] = output
            all_outputs.append(output)

            if output.exit_code != 0:
                logger.error(f"Step '{step.id}' failed with exit code {output.exit_code}: {output.stderr}")
                # Halt execution on step failure
                return all_outputs

            logger.info(f"Step '{step.id}' completed successfully.")

        return all_outputs 