import logging
import re
from typing import Dict, Any, List

from ..schemas.messages import A2APlanState, A2ARequest, ParamsContent
from ..agents.agent_registry import AgentRegistry

try:
    from a2a.client import post as a2a_post
    from a2a.model import Response as A2AResponse
    A2A_SDK_AVAILABLE = True
except ImportError:
    A2A_SDK_AVAILABLE = False


class A2AExecutor:
    """
    Executes a plan by making A2A calls to the appropriate agents.
    Now refactored to use a2a-sdk for client-side requests.
    """
    def __init__(self, agent_registry: AgentRegistry = None):
        self._agent_registry = agent_registry or AgentRegistry()

    def _resolve_dependencies(self, parameters: Dict[str, Any], step_outputs: List[tuple]) -> Dict[str, Any]:
        """Resolves dependencies in parameters using outputs from previous steps."""
        resolved_params = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                match = re.match(r"\{\{steps\[(\d+)\]\.(.*)\}\}", value)
                if match:
                    step_num = int(match.group(1))
                    json_path = match.group(2)
                    if step_num < len(step_outputs):
                        try:
                            # a2a_executor_node appends a tuple, result is at index 2
                            previous_result = step_outputs[step_num][2]
                            # A simple path resolver for now. For e.g. "contents[0].data.df_id"
                            # A more robust solution would use a proper JSONPath library.
                            temp_val = previous_result
                            for part in json_path.replace(']', '').replace('[', '.').split('.'):
                                if part.isdigit():
                                    temp_val = temp_val[int(part)]
                                else:
                                    temp_val = temp_val[part]
                            resolved_params[key] = temp_val
                        except (KeyError, IndexError, TypeError) as e:
                             raise ValueError(f"Could not resolve dependency from step {step_num} with path '{json_path}': {e}")
                    else:
                        raise ValueError(f"Output for dependency step {step_num} not found.")
                else:
                    resolved_params[key] = value
            else:
                resolved_params[key] = value
        return resolved_params
    
    async def _a2a_call(self, agent_name: str, request: A2ARequest) -> Dict[str, Any]:
        """Performs the HTTP call to the agent server using a2a-sdk."""
        if not A2A_SDK_AVAILABLE:
            raise RuntimeError("a2a-sdk is not installed. Cannot make A2A calls.")

        agent_url = self._agent_registry.get_agent_url(agent_name)
        if not agent_url:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
        
        # a2a.client.post handles the async call and returns a pydantic Response object
        response: A2AResponse = await a2a_post(f"{agent_url}/process", request=request, timeout=60.0)
        
        # Return as a dictionary to maintain compatibility with the rest of the executor logic
        return response.model_dump(by_alias=True)

    async def execute(self, state: A2APlanState) -> A2APlanState:
        """Executes the entire plan sequentially."""
        logging.info("Starting A2A plan execution.")
        
        step_outputs: List[tuple] = []

        for i, step in enumerate(state.plan):
            state.current_step = i
            agent_name = step.get("agent_name")
            action = step.get("action")
            parameters = step.get("parameters", {})

            logging.info(f"Executing Step {i + 1}/{len(state.plan)}: Agent '{agent_name}', Action '{action}'")

            try:
                resolved_params = self._resolve_dependencies(parameters, step_outputs)
                
                request = A2ARequest(
                    action=action,
                    contents=[ParamsContent(data=resolved_params)]
                )
                
                result = await self._a2a_call(agent_name, request)
                
                if result.get("status") != "success":
                    state.error_message = f"Step {i+1} failed: {result.get('message', 'Unknown error')}"
                    logging.error(state.error_message)
                    return state

                logging.info(f"Step {i + 1} successful.")
                step_outputs.append((agent_name, action, result))
                
            except Exception as e:
                state.error_message = f"An unexpected error occurred during step {i+1}: {e}"
                logging.error(state.error_message, exc_info=True)
                return state

        state.previous_steps = step_outputs
        state.current_step = len(state.plan)
        logging.info("A2A plan execution completed successfully.")
        return state 