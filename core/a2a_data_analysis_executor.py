import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime

from core.callbacks.progress_stream import progress_stream_manager

logger = logging.getLogger(__name__)

class A2ADataAnalysisExecutor:
    """데이터 분석용 A2A 실행기"""
    
    def __init__(self):
        self.agent_configs = {
            'pandas_data_analyst': {
                'url': 'http://localhost:8200',
                'capabilities': ['data_structure', 'descriptive_stats', 'correlation_analysis', 'analyze_data']
            },
            'sql_data_analyst': {
                'url': 'http://localhost:8201', 
                'capabilities': ['sql_queries', 'database_analysis', 'analyze_data']
            },
            'data_visualization': {
                'url': 'http://localhost:8202',
                'capabilities': ['charts', 'plots', 'interactive_viz', 'analyze_data']
            },
            'eda_tools': {
                'url': 'http://localhost:8203',
                'capabilities': ['outlier_detection', 'distribution_analysis', 'analyze_data']
            },
            'feature_engineering': {
                'url': 'http://localhost:8204',
                'capabilities': ['feature_creation', 'feature_selection', 'analyze_data']
            },
            'data_cleaning': {
                'url': 'http://localhost:8205',
                'capabilities': ['missing_values', 'data_quality', 'analyze_data']
            }
        }
        self.timeout = 300  # 5분 타임아웃
    
    async def execute(self, plan_state: dict) -> dict:
        """계획을 순차적으로 실행"""
        logger.info(f"Starting execution of plan with {len(plan_state.get('plan', []))} steps")
        
        step_outputs = {}
        execution_start_time = datetime.now()
        
        try:
            for i, step in enumerate(plan_state.get("plan", []), 1):
                agent_name = step.get("agent_name")
                agent_config = self.agent_configs.get(agent_name)
                
                if not agent_config:
                    await self.emit_error(i, agent_name, f"Unknown agent: {agent_name}")
                    continue
                
                logger.info(f"Executing step {i}: {agent_name}")
                
                try:
                    # 에이전트 시작 알림
                    await progress_stream_manager.stream_update({
                        "event_type": "agent_start",
                        "data": {"step": i, "agent_name": agent_name}
                    })
                    
                    # A2A 호출 실행
                    result = await self.call_agent(agent_config, step, i)
                    
                    if result and result.get('success'):
                        step_outputs[i] = result
                        # 성공 알림
                        await progress_stream_manager.stream_update({
                            "event_type": "agent_end",
                            "data": {
                                "step": i, 
                                "agent_name": agent_name,
                                "output": result
                            }
                        })
                        logger.info(f"Step {i} completed successfully")
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'No response'
                        await self.emit_error(i, agent_name, error_msg)
                        
                except Exception as e:
                    logger.error(f"Error in step {i}: {str(e)}")
                    await self.emit_error(i, agent_name, str(e))
            
            execution_end_time = datetime.now()
            execution_duration = (execution_end_time - execution_start_time).total_seconds()
            
            logger.info(f"Plan execution completed in {execution_duration:.2f} seconds")
            
            return {
                "step_outputs": step_outputs,
                "execution_time": execution_duration,
                "total_steps": len(plan_state.get("plan", [])),
                "successful_steps": len(step_outputs)
            }
            
        except Exception as e:
            logger.error(f"Fatal error during plan execution: {str(e)}")
            return {
                "error": str(e),
                "step_outputs": step_outputs
            }
    
    async def call_agent(self, agent_config: dict, step: dict, step_num: int) -> Optional[dict]:
        """개별 에이전트 호출"""
        agent_url = agent_config['url']
        
        # Task ID 생성
        task_id = str(uuid.uuid4())
        
        # 요청 페이로드 구성 (A2A SDK v0.2.9 호환)
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": task_id,
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": step.get("parameters", {}).get("user_instructions", "")
                        },
                        {
                            "type": "data", 
                            "data": {
                                "data_id": step.get("parameters", {}).get("data_id", ""),
                                "skill_name": step.get("skill_name", "analyze_data")
                            }
                        }
                    ]
                }
            },
            "id": step_num
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(f"Calling agent at {agent_url} with payload: {payload}")
                
                response = await client.post(agent_url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                logger.debug(f"Agent response: {result}")
                
                if "result" in result:
                    # JSON-RPC 성공 응답
                    return {
                        "success": True,
                        "content": result["result"],
                        "agent": step.get("agent_name"),
                        "step": step_num
                    }
                elif "error" in result:
                    # JSON-RPC 에러 응답
                    return {
                        "success": False,
                        "error": result["error"].get("message", "Unknown RPC error"),
                        "agent": step.get("agent_name"),
                        "step": step_num
                    }
                else:
                    return {
                        "success": False,
                        "error": "Invalid response format",
                        "agent": step.get("agent_name"),
                        "step": step_num
                    }
                    
        except httpx.TimeoutException:
            logger.error(f"Timeout calling agent {agent_url}")
            return {
                "success": False,
                "error": f"Timeout after {self.timeout} seconds",
                "agent": step.get("agent_name"),
                "step": step_num
            }
        except httpx.RequestError as e:
            logger.error(f"Request error calling agent {agent_url}: {str(e)}")
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "agent": step.get("agent_name"),
                "step": step_num
            }
        except Exception as e:
            logger.error(f"Unexpected error calling agent {agent_url}: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "agent": step.get("agent_name"),
                "step": step_num
            }
    
    async def emit_error(self, step: int, agent_name: str, error: str):
        """에러 이벤트 발생"""
        logger.error(f"Step {step} error in {agent_name}: {error}")
        await progress_stream_manager.stream_update({
            "event_type": "agent_error",
            "data": {
                "step": step,
                "agent_name": agent_name,
                "error": error
            }
        })
    
    async def check_agent_health(self, agent_url: str) -> bool:
        """에이전트 상태 확인"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{agent_url}/.well-known/agent.json")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_available_agents(self) -> List[str]:
        """사용 가능한 에이전트 목록 반환"""
        available_agents = []
        
        for agent_name, config in self.agent_configs.items():
            if await self.check_agent_health(config['url']):
                available_agents.append(agent_name)
            else:
                logger.warning(f"Agent {agent_name} at {config['url']} is not available")
        
        return available_agents 