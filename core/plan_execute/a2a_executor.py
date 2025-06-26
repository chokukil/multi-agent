import asyncio
import logging
import httpx
import uuid
from typing import Dict, Any, Union

# A2A SDK 공식 클라이언트와 표준 타입들 사용 (실제 사용 가능한 API)
import uuid
from a2a.client import A2AClient
from a2a.types import (
    Message, MessageSendParams, TaskState, Task,
    SendMessageRequest, TextPart, Role
)
from a2a.utils.message import new_agent_text_message

from core.callbacks.progress_stream import progress_stream_manager
from core.utils.logging import log_a2a_request, log_a2a_response
from core.direct_analysis import direct_analysis_engine

logger = logging.getLogger(__name__)

class A2AExecutor:
    """
    A2A SDK 표준을 완전히 준수하는 실행기
    공식 A2A 클라이언트와 프로토콜을 사용하여 에이전트와 통신
    """

    def __init__(self):
        self.progress_stream = progress_stream_manager
        logger.info("🔧 A2AExecutor initialized with standard SDK")

    async def _create_agent_client_context(self, agent_url: str):
        """A2A 클라이언트와 HTTP 클라이언트를 함께 관리하는 컨텍스트 생성"""
        logger.info(f"🌐 Connecting to A2A agent at: {agent_url}")
        
        try:
            # 서버 연결 테스트 (선택사항)
            async with httpx.AsyncClient(timeout=5.0) as test_client:
                logger.debug(f"🔍 Testing server connectivity")
                # A2A 표준 agent card 엔드포인트 확인
                card_response = await test_client.get(f"{agent_url}/.well-known/agent.json", timeout=5.0)
                logger.debug(f"✅ Agent card accessible: {card_response.status_code}")
                log_a2a_response(card_response.status_code, {"agent_card_check": "passed"})
        
        except Exception as health_error:
            logger.warning(f"⚠️ Server connectivity test failed: {health_error}")
            log_a2a_response(0, error=f"Connectivity test failed: {health_error}")
            # 계속 진행 - 실제 연결에서 실패할 수 있지만 시도해볼 가치가 있음

        # HTTP 클라이언트와 A2A 클라이언트를 함께 반환 (타임아웃 개선)
        httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,  # 연결 타임아웃
                read=30.0,    # 읽기 타임아웃 (단축)
                write=10.0,   # 쓰기 타임아웃  
                pool=5.0      # 풀 타임아웃
            )
        )
        try:
            log_a2a_request("GET", f"{agent_url}/.well-known/agent.json")
            client = await A2AClient.get_client_from_agent_card_url(httpx_client, agent_url)
            logger.info("✅ A2A client created successfully")
            log_a2a_response(200, {"client_creation": "success"})
            return httpx_client, client
            
        except Exception as client_error:
            await httpx_client.aclose()  # 실패 시 HTTP 클라이언트 정리
            logger.error(f"❌ Failed to create A2A client: {client_error}")
            log_a2a_response(0, error=f"Client creation failed: {client_error}")
            raise

    async def _send_message(self, httpx_client: httpx.AsyncClient, client: A2AClient, message: Message, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 메시지 전송 (표준 방식 - 클라이언트 생명주기 관리)"""
        logger.info("📤 Sending message via A2A protocol...")
        
        try:
            # A2A 표준 메시지 전송 파라미터 생성
            send_params = MessageSendParams(message=message)
            request_id = str(uuid.uuid4())
            
            request = SendMessageRequest(
                id=request_id,
                params=send_params
            )
            
            log_a2a_request("POST", "A2A Client send_message", {"request_id": request_id})
            response = await client.send_message(request)
            
            if response:
                logger.info("✅ Message sent successfully")
                log_a2a_response(200, {"message_sent": "success"})
                return self._process_response(response, step_info)
            else:
                logger.error("❌ No response received")
                log_a2a_response(0, error="No response received")
                raise RuntimeError("No response from agent")
                
        except Exception as e:
            logger.error(f"❌ Message sending failed: {e}", exc_info=True)
            log_a2a_response(0, error=f"Message sending failed: {e}")
            raise

    def _process_response(self, response: Any, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 응답 처리 (표준 방식)"""
        logger.info(f"🎯 Processing A2A response: {type(response)}")
        logger.info(f"🔍 Response content: {response}")
        
        try:
            # A2A 응답 구조 확인
            if hasattr(response, 'root'):
                result = response.root
                logger.info(f"✅ A2A response structure found: {type(result)}")
                logger.info(f"🔍 Result attributes: {dir(result)}")
                
                if hasattr(result, 'result'):
                    # 성공 응답
                    actual_result = result.result
                    logger.info(f"📊 Response type: {type(actual_result)}")
                    logger.info(f"🔍 Actual result content: {actual_result}")
                    
                    if isinstance(actual_result, Message):
                        # 메시지 응답 처리
                        content = ""
                        if actual_result.parts:
                            for part in actual_result.parts:
                                # Part 구조: Part(root=TextPart(...))
                                if hasattr(part, 'root') and hasattr(part.root, 'text') and part.root.text:
                                    content += part.root.text
                                elif hasattr(part, 'text') and part.text:  # fallback
                                    content += part.text
                        
                        logger.info(f"📝 Extracted message content length: {len(content)}")
                        
                        return {
                            "messageId": getattr(actual_result, 'messageId', str(uuid.uuid4())),
                            "content": content,
                            "response_type": "message",
                            "success": True
                        }
                    
                    elif hasattr(actual_result, 'status'):
                        # Task 응답 처리
                        task = actual_result
                        return {
                            "task_id": task.id,
                            "status": task.status.state,
                            "response_type": "task", 
                            "success": True
                        }
                    
                    else:
                        # 기타 성공 응답
                        return {
                            "result": str(actual_result),
                            "response_type": "generic",
                            "success": True
                        }
                
                elif hasattr(result, 'error'):
                    # 에러 응답
                    error = result.error
                    return {
                        "error": str(error),
                        "response_type": "error",
                        "success": False
                    }
            
            # 응답 구조를 파악할 수 없는 경우
            logger.warning(f"⚠️ Unknown response structure: {response}")
            return {
                "result": str(response),
                "response_type": "unknown",
                "success": True
            }
                
        except Exception as e:
            logger.error(f"❌ Response processing failed: {e}", exc_info=True)
            return {
                "error": f"Response processing failed: {str(e)}",
                "response_type": "error",
                "success": False
            }

    async def execute(self, state: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """A2A 표준 프로토콜을 사용한 플랜 실행"""
        plan_steps = state.get("plan", [])
        step_outputs = {}

        logger.info("🎬 A2A EXECUTOR STARTING (Standard SDK)")
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
            logger.info(f"   - Data ID: {data_id}")

            await self.progress_stream.stream_update({
                "event_type": "agent_start", 
                "data": step_info
            })

            try:
                # A2A 에이전트 URL을 config에서 가져오기
                from config import AGENT_SERVERS
                
                # agent_name을 config key에 매핑
                agent_mapping = {
                    "pandas_data_analyst": "pandas_analyst",
                    "EDA": "pandas_analyst",  # EDA Copilot용
                }
                
                config_key = agent_mapping.get(agent_name, agent_name)
                agent_config = AGENT_SERVERS.get(config_key, {})
                agent_url = agent_config.get("url", "http://localhost:10001")
                
                logger.info(f"🌐 Agent URL for '{agent_name}' -> '{config_key}': {agent_url}")
                
                # A2A 클라이언트와 HTTP 클라이언트 생성
                httpx_client, client = await self._create_agent_client_context(agent_url)
                
                try:
                    # A2A 표준 메시지 생성
                    skill_instruction = f"""
Please execute the '{skill_name}' skill with the following request:

User Request: {user_prompt}
Data ID: {data_id}

Please analyze the dataset with ID '{data_id}' based on the user's instructions.
                    """.strip()
                    
                    # A2A 표준 사용자 메시지 생성
                    user_message = Message(
                        messageId=str(uuid.uuid4()),
                        role=Role.user,
                        parts=[TextPart(text=skill_instruction)]
                    )
                    logger.info(f"📝 Created A2A message with {len(user_message.parts)} parts")
                    
                    # 메시지 전송 및 응답 처리
                    result = await self._send_message(httpx_client, client, user_message, step_info)
                    logger.info(f"📥 A2A Response received: {type(result)}")
                    logger.info(f"📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    
                    # 응답 내용 로깅
                    if isinstance(result, dict):
                        if result.get("content"):
                            content_preview = str(result["content"])[:200]
                            logger.info(f"✅ Content received ({len(str(result['content']))} chars): {content_preview}...")
                        else:
                            logger.warning("⚠️ No content in response")
                    
                    step_outputs[step_info["step"]] = result
                    
                    # 완료 알림
                    await self.progress_stream.stream_update({
                        "event_type": "agent_end",
                        "data": {**step_info, "output": result}
                    })
                    
                    logger.info("✅ Step completed successfully")
                    
                finally:
                    # HTTP 클라이언트 정리
                    await httpx_client.aclose()
                    logger.debug("🧹 HTTP client closed")
                
            except Exception as e:
                logger.error(f"💥 A2A Step execution failed: {e}", exc_info=True)
                logger.info("🔄 Switching to Direct Analysis Engine as fallback...")
                
                # 🆕 A2A 실패 시 직접 분석 엔진으로 대체
                try:
                    fallback_result = direct_analysis_engine.analyze_with_fallback(user_prompt, data_id)
                    step_outputs[step_info["step"]] = fallback_result
                    
                    # 성공 알림 (대체 분석 완료)
                    await self.progress_stream.stream_update({
                        "event_type": "agent_end",
                        "data": {**step_info, "output": fallback_result}
                    })
                    
                    logger.info("✅ Direct Analysis completed successfully as fallback")
                    
                except Exception as fallback_error:
                    logger.error(f"💥 Fallback analysis also failed: {fallback_error}", exc_info=True)
                    step_outputs[step_info["step"]] = {
                        "error": f"A2A failed: {str(e)}\nFallback failed: {str(fallback_error)}",
                        "response_type": "error",
                        "success": False
                    }
                    
                    await self.progress_stream.stream_update({
                        "event_type": "agent_error",
                        "data": {**step_info, "error": str(fallback_error)}
                    })

        logger.info("🎬 A2A EXECUTOR COMPLETED")
        return {"step_outputs": step_outputs, "status": "completed"} 