import logging
import httpx
import re
from typing import Dict, Any, Coroutine, Callable, List

from ..schemas.messages import A2APlanState, A2ARequest, ParamsContent
from ..agents.agent_registry import AgentRegistry

class A2AExecutor:
    """
    Executes a plan by making A2A calls to the appropriate agents.
    """
    def __init__(self, agent_registry: AgentRegistry = None):
        self._agent_registry = agent_registry or AgentRegistry()

    def _resolve_dependencies(self, parameters: Dict[str, Any], step_outputs: Dict[int, Any]) -> Dict[str, Any]:
        """Resolves dependencies in parameters using outputs from previous steps."""
        resolved_params = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                # Simple substitution for {{steps[index].output.contents[0].data.df_id}}
                match = re.match(r"\{\{steps\[(\d+)\]\..*df_id\}\}", value)
                if match:
                    step_num = int(match.group(1))
                    if step_num < len(step_outputs):
                        # Assuming the output is from a previous step in the format (agent, skill, result_dict)
                        # and result_dict has a specific structure.
                        # This is a simplification. A more robust solution would use JSONPath.
                        try:
                            # a2a_executor_node appends a tuple, result is at index 2
                            previous_result = step_outputs[step_num][2]
                            df_id = previous_result['contents'][0]['data']['df_id']
                            resolved_params[key] = df_id
                        except (KeyError, IndexError) as e:
                             raise ValueError(f"Could not resolve dependency from step {step_num}: {e}")
                    else:
                        raise ValueError(f"Output for dependency step {step_num} not found.")
                else:
                    resolved_params[key] = value
            else:
                resolved_params[key] = value
        return resolved_params
    
    async def _a2a_call(self, agent_name: str, request_body: dict) -> Dict[str, Any]:
        """Performs the actual HTTP call to the agent server."""
        agent_url = self._agent_registry.get_agent_url(agent_name)
        if not agent_url:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{agent_url}/process", json=request_body, timeout=60.0)
            response.raise_for_status()
            return response.json()

    async def execute(self, state: A2APlanState) -> A2APlanState:
        """Executes the entire plan sequentially."""
        logging.info("Starting A2A plan execution.")
        
        step_outputs = []

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
                
                result = await self._a2a_call(agent_name, request.model_dump(by_alias=True))
                
                if result.get("status") != "success":
                    state.error_message = f"Step {i+1} failed: {result.get('message', 'Unknown error')}"
                    logging.error(state.error_message)
                    return state

                logging.info(f"Step {i + 1} successful. Result: {result}")
                step_outputs.append((agent_name, action, result))
                
            except Exception as e:
                state.error_message = f"An unexpected error occurred during step {i+1}: {e}"
                logging.error(state.error_message, exc_info=True)
                return state

        state.previous_steps = step_outputs
        state.current_step = len(state.plan)
        logging.info("A2A plan execution completed successfully.")
        return state 