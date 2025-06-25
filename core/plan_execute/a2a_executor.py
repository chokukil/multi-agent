import asyncio
import logging
import httpx
import uuid
from typing import Dict, Any, Union

from a2a.client import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams, TaskState, GetTaskRequest, TaskQueryParams
from a2a.utils.message import new_agent_text_message

from core.callbacks.progress_stream import progress_stream_manager

logger = logging.getLogger(__name__)

class A2AExecutor:
    """
    Executes a plan by making A2A calls to agent skills using the a2a-sdk.
    """

    def __init__(self):
        self.progress_stream = progress_stream_manager

    async def _poll_task_status(self, client: A2AClient, task_id: str, step_info: Dict[str, Any]):
        """Polls the agent server for task completion."""
        while True:
            await asyncio.sleep(1)  # Polling interval
            try:
                get_task_req = GetTaskRequest(
                    id=str(uuid.uuid4()),  # Required JSON-RPC id field
                    params=TaskQueryParams(id=task_id)
                )
                response = await client.get_task(get_task_req)
                
                if not response or not hasattr(response.root, "result"):
                    # Handle cases where the response is not as expected
                    logger.warning(f"Polling for task {task_id} returned an invalid response.")
                    continue

                task = response.root.result
                
                if task.status.state in (TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected):
                    logger.info(f"Task {task_id} finished with state: {task.status.state}")
                    return task
            except Exception as e:
                logger.error(f"Error polling task {task_id}: {e}", exc_info=True)
                # On polling error, we assume the task has failed to avoid infinite loops
                return None

    async def execute(self, state: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Executes a plan from a state dictionary and streams progress."""
        plan_steps = state.get("plan", [])
        step_outputs = {}

        logger.info("🎬 A2A EXECUTOR STARTING")
        logger.info(f"📋 Plan has {len(plan_steps)} steps")

        for step_info in plan_steps:
            agent_name = step_info.get("agent_name")
            skill_name = step_info.get("skill_name")
            params = step_info.get("parameters", {})
            user_prompt = params.get("user_instructions", "No prompt provided.")
            data_id = params.get("data_id")

            logger.info("="*50)
            logger.info(f"🎯 EXECUTING STEP: {step_info.get('step', '?')}")
            logger.info(f"   - Agent: {agent_name}")
            logger.info(f"   - Skill: {skill_name}")
            logger.info(f"   - User prompt: {repr(user_prompt[:100])}...")
            logger.info(f"   - Data ID: {data_id}")
            logger.info(f"   - All params: {params}")

            await self.progress_stream.stream_update({"event_type": "agent_start", "data": step_info})

            try:
                # Agent URL should be defined in a config, not hardcoded.
                # For now, we'll derive it from the agent name.
                agent_url = f"http://localhost:10001"  # TODO: Get from a config based on agent_name
                logger.info(f"🌐 Connecting to agent at: {agent_url}")

                async with httpx.AsyncClient(timeout=timeout) as httpx_client:
                    # 1. Get client from agent card
                    logger.info("📋 Fetching agent card...")
                    client = await A2AClient.get_client_from_agent_card_url(httpx_client, agent_url)
                    logger.info("✅ Agent card fetched successfully")

                    # 2. Create A2A message according to the official protocol
                    # The message should contain the skill request in natural language
                    # The server will parse this and route to the appropriate skill
                    skill_instruction = f"""
Please execute the '{skill_name}' skill with the following request:

User Request: {user_prompt}
Data ID: {data_id}

Please analyze the dataset with ID '{data_id}' based on the user's instructions.
                    """.strip()
                    
                    logger.info("📝 CREATING A2A MESSAGE:")
                    logger.info(f"   - Skill instruction: {repr(skill_instruction)}")
                    
                    user_message = new_agent_text_message(skill_instruction)
                    logger.info(f"   - Message type: {type(user_message)}")
                    logger.info(f"   - Message parts count: {len(user_message.parts) if hasattr(user_message, 'parts') else 'N/A'}")
                    
                    # 3. Create MessageSendParams with ONLY the message field (per A2A spec)
                    send_params = MessageSendParams(
                        message=user_message
                    )
                    logger.info(f"   - Send params type: {type(send_params)}")
                    
                    # 4. Create SendMessageRequest with required JSON-RPC id field
                    request_id = str(uuid.uuid4())
                    request = SendMessageRequest(
                        id=request_id,  # Required JSON-RPC id field
                        params=send_params
                    )
                    logger.info(f"   - Request ID: {request_id}")
                    logger.info(f"   - Request type: {type(request)}")
                    
                    # 5. Send the message using the correct A2A protocol
                    # Using message/send (NOT tasks/send) as per A2A v0.2.0 specification
                    logger.info("📤 SENDING MESSAGE via A2A protocol...")
                    response = await client.send_message(request)
                    logger.info(f"📥 RESPONSE RECEIVED: {type(response)}")

                    if not response or not hasattr(response.root, "result"):
                        logger.error("❌ No valid response received from agent")
                        raise ConnectionError("Failed to initiate task with agent.")

                    result = response.root.result
                    logger.info(f"🎯 RESULT TYPE: {type(result)}")
                    
                    # 6. Handle both Message and Task response types per A2A specification
                    if hasattr(result, 'kind'):
                        result_kind = result.kind
                        logger.info(f"✅ Result has 'kind' attribute: {result_kind}")
                    else:
                        # Fallback: try to determine type from available attributes
                        result_kind = "task" if hasattr(result, 'status') else "message"
                        logger.info(f"🔍 Determined result kind from attributes: {result_kind}")
                        logger.info(f"   - Has 'status': {hasattr(result, 'status')}")
                        logger.info(f"   - Has 'parts': {hasattr(result, 'parts')}")
                        logger.info(f"   - Has 'messageId': {hasattr(result, 'messageId')}")
                    
                    if result_kind == "message":
                        # Direct message response - immediate completion
                        logger.info("📨 PROCESSING DIRECT MESSAGE RESPONSE")
                        step_outputs[step_info["step"]] = {
                            "messageId": getattr(result, 'messageId', str(uuid.uuid4())),
                            "parts": getattr(result, 'parts', []),
                            "response_type": "direct_message"
                        }
                        logger.info(f"   - Output created: {step_outputs[step_info['step']]}")
                        
                        await self.progress_stream.stream_update({
                            "event_type": "agent_end", 
                            "data": {**step_info, "output": step_outputs[step_info["step"]]}
                        })
                        logger.info("✅ Direct message response processed successfully")
                        
                    elif result_kind == "task":
                        # Task-based response - may need polling
                        logger.info("📋 PROCESSING TASK RESPONSE")
                        task = result
                        task_id = task.id
                        logger.info(f"   - Task ID: {task_id}")
                        logger.info(f"   - Task status: {task.status.state}")
                        
                        # Check if task is already completed
                        if task.status.state == TaskState.completed:
                            logger.info("✅ Task already completed")
                            final_artifact = task.artifacts[0] if task.artifacts else {}
                            step_outputs[step_info["step"]] = final_artifact
                            await self.progress_stream.stream_update({
                                "event_type": "agent_end", 
                                "data": {**step_info, "output": final_artifact}
                            })
                        else:
                            # 6. Poll for task completion
                            logger.info("⏳ Task not completed, starting polling...")
                            final_task = await self._poll_task_status(client, task_id, step_info)

                            if final_task and final_task.status.state == TaskState.completed:
                                logger.info("✅ Task completed after polling")
                                final_artifact = final_task.artifacts[0] if final_task.artifacts else {}
                                step_outputs[step_info["step"]] = final_artifact
                                await self.progress_stream.stream_update({
                                    "event_type": "agent_end", 
                                    "data": {**step_info, "output": final_artifact}
                                })
                            else:
                                error_msg = f"Task failed with state: {final_task.status.state if final_task else 'Polling Failed'}"
                                logger.error(f"❌ {error_msg}")
                                raise RuntimeError(error_msg)
                    else:
                        # Unknown response type
                        logger.error(f"❌ Unknown response type: {result_kind}")
                        raise RuntimeError(f"Unknown response type: {result_kind}")

            except Exception as e:
                logger.error("="*50)
                logger.error(f"💥 ERROR executing step for agent '{agent_name}': {e}")
                logger.error("="*50, exc_info=True)
                error_message = str(e)
                await self.progress_stream.stream_update({
                    "event_type": "agent_error",
                    "data": {**step_info, "error_message": error_message}
                })
                state["error"] = f"Step failed: {error_message}"
                return state

        logger.info("🎉 A2A EXECUTOR COMPLETED SUCCESSFULLY")
        logger.info(f"📊 Generated {len(step_outputs)} outputs")
        state["outputs"] = step_outputs
        return state 