import asyncio
import logging
import httpx
import uuid
from typing import Dict, Any, Union

from a2a.client import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams, TaskState, GetTaskRequest, TaskQueryParams
from a2a.utils.message import new_agent_text_message

from core.callbacks.progress_stream import progress_stream_manager
from core.utils.logging import log_a2a_request, log_a2a_response

logger = logging.getLogger(__name__)

class A2AExecutor:
    """
    Executes a plan by making A2A calls to agent skills using the a2a-sdk.
    """

    def __init__(self):
        self.progress_stream = progress_stream_manager

    async def _poll_task_status(self, client: A2AClient, task_id: str, step_info: Dict[str, Any]):
        """Polls the agent server for task completion."""
        logger.debug(f"🔄 Starting task polling for task_id: {task_id}")
        poll_count = 0
        max_polls = 120  # 2 minutes with 1-second intervals
        
        while poll_count < max_polls:
            poll_count += 1
            await asyncio.sleep(1)  # Polling interval
            
            try:
                logger.debug(f"📊 Poll #{poll_count}/{max_polls} for task {task_id}")
                get_task_req = GetTaskRequest(
                    id=str(uuid.uuid4()),  # Required JSON-RPC id field
                    params=TaskQueryParams(id=task_id)
                )
                
                log_a2a_request("GET", f"/tasks/{task_id}", {"task_id": task_id})
                response = await client.get_task(get_task_req)
                
                if not response or not hasattr(response.root, "result"):
                    logger.warning(f"⚠️ Poll #{poll_count}: Invalid response for task {task_id}")
                    log_a2a_response(0, error="Invalid response structure")
                    continue

                task = response.root.result
                logger.debug(f"📋 Poll #{poll_count}: Task {task_id} status = {task.status.state}")
                
                if task.status.state in (TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected):
                    logger.info(f"🏁 Task {task_id} finished with state: {task.status.state} after {poll_count} polls")
                    log_a2a_response(200, {"task_id": task_id, "status": task.status.state})
                    return task
                    
            except Exception as e:
                logger.error(f"💥 Poll #{poll_count} error for task {task_id}: {e}", exc_info=True)
                log_a2a_response(0, error=str(e))
                # Continue polling on error, don't fail immediately
                
        logger.error(f"⏰ Task {task_id} polling timeout after {max_polls} attempts")
        return None

    async def execute(self, state: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Executes a plan from a state dictionary and streams progress."""
        plan_steps = state.get("plan", [])
        step_outputs = {}

        logger.info("🎬 A2A EXECUTOR STARTING")
        logger.info(f"📋 Plan has {len(plan_steps)} steps")
        logger.debug(f"🔍 Full plan: {plan_steps}")

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
            logger.debug(f"   - All params: {params}")
            logger.debug(f"   - Step info: {step_info}")

            await self.progress_stream.stream_update({"event_type": "agent_start", "data": step_info})

            try:
                # Agent URL should be defined in a config, not hardcoded.
                # For now, we'll derive it from the agent name.
                agent_url = f"http://localhost:10001"  # TODO: Get from a config based on agent_name
                logger.info(f"🌐 Connecting to agent at: {agent_url}")

                # First, let's test if the server is reachable
                try:
                    async with httpx.AsyncClient(timeout=5.0) as test_client:
                        logger.debug(f"🔍 Testing server connectivity to {agent_url}")
                        test_response = await test_client.get(f"{agent_url}/health", timeout=5.0)
                        logger.debug(f"✅ Server health check: {test_response.status_code}")
                        log_a2a_response(test_response.status_code, {"health_check": "passed"})
                except Exception as health_error:
                    logger.error(f"❌ Server health check failed: {health_error}")
                    log_a2a_response(0, error=f"Health check failed: {health_error}")

                async with httpx.AsyncClient(timeout=timeout) as httpx_client:
                    # 1. Get client from agent card
                    logger.info("📋 Fetching agent card...")
                    try:
                        log_a2a_request("GET", f"{agent_url}/agent.json")
                        client = await A2AClient.get_client_from_agent_card_url(httpx_client, agent_url)
                        logger.info("✅ Agent card fetched successfully")
                        log_a2a_response(200, {"agent_card": "fetched"})
                    except Exception as card_error:
                        logger.error(f"❌ Failed to fetch agent card: {card_error}")
                        log_a2a_response(0, error=f"Agent card fetch failed: {card_error}")
                        raise

                    # 2. Create A2A message according to the official protocol
                    skill_instruction = f"""
Please execute the '{skill_name}' skill with the following request:

User Request: {user_prompt}
Data ID: {data_id}

Please analyze the dataset with ID '{data_id}' based on the user's instructions.
                    """.strip()
                    
                    logger.info("📝 CREATING A2A MESSAGE:")
                    logger.debug(f"   - Skill instruction: {repr(skill_instruction)}")
                    
                    user_message = new_agent_text_message(skill_instruction)
                    logger.debug(f"   - Message type: {type(user_message)}")
                    logger.debug(f"   - Message parts count: {len(user_message.parts) if hasattr(user_message, 'parts') else 'N/A'}")
                    
                    # 3. Create MessageSendParams with ONLY the message field (per A2A spec)
                    send_params = MessageSendParams(
                        message=user_message
                    )
                    logger.debug(f"   - Send params type: {type(send_params)}")
                    
                    # 4. Create SendMessageRequest with required JSON-RPC id field
                    request_id = str(uuid.uuid4())
                    request = SendMessageRequest(
                        id=request_id,  # Required JSON-RPC id field
                        params=send_params
                    )
                    logger.info(f"   - Request ID: {request_id}")
                    logger.debug(f"   - Request type: {type(request)}")
                    
                    # 5. Send the message using the correct A2A protocol
                    logger.info("📤 SENDING MESSAGE via A2A protocol...")
                    try:
                        log_a2a_request("POST", f"{agent_url}/message/send", {
                            "request_id": request_id,
                            "skill_instruction": skill_instruction[:200] + "..." if len(skill_instruction) > 200 else skill_instruction
                        })
                        
                        response = await client.send_message(request)
                        logger.info(f"📥 RESPONSE RECEIVED: {type(response)}")
                        log_a2a_response(200, {"response_type": str(type(response))})
                        
                    except httpx.HTTPStatusError as http_error:
                        logger.error(f"❌ HTTP Error: {http_error.response.status_code} - {http_error.response.text}")
                        log_a2a_response(http_error.response.status_code, error=http_error.response.text)
                        raise
                    except Exception as send_error:
                        logger.error(f"❌ Send message failed: {send_error}")
                        log_a2a_response(0, error=str(send_error))
                        raise

                    if not response or not hasattr(response.root, "result"):
                        logger.error("❌ No valid response received from agent")
                        log_a2a_response(0, error="No valid response structure")
                        raise ConnectionError("Failed to initiate task with agent.")

                    result = response.root.result
                    logger.info(f"🎯 RESULT TYPE: {type(result)}")
                    logger.debug(f"🔍 Result attributes: {dir(result)}")
                    
                    # 6. Handle both Message and Task response types per A2A specification
                    if hasattr(result, 'kind'):
                        result_kind = result.kind
                        logger.info(f"✅ Result has 'kind' attribute: {result_kind}")
                    else:
                        # Fallback: try to determine type from available attributes
                        result_kind = "task" if hasattr(result, 'status') else "message"
                        logger.info(f"🔍 Determined result kind from attributes: {result_kind}")
                        logger.debug(f"   - Has 'status': {hasattr(result, 'status')}")
                        logger.debug(f"   - Has 'parts': {hasattr(result, 'parts')}")
                        logger.debug(f"   - Has 'messageId': {hasattr(result, 'messageId')}")
                    
                    if result_kind == "message":
                        # Direct message response - immediate completion
                        logger.info("📨 PROCESSING DIRECT MESSAGE RESPONSE")
                        step_outputs[step_info["step"]] = {
                            "messageId": getattr(result, 'messageId', str(uuid.uuid4())),
                            "parts": getattr(result, 'parts', []),
                            "response_type": "direct_message"
                        }
                        logger.debug(f"   - Output created: {step_outputs[step_info['step']]}")
                        
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
                        logger.debug(f"   - Task details: {task}")
                        
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
                logger.error(f"💥 Error type: {type(e)}")
                logger.error(f"💥 Error args: {e.args}")
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