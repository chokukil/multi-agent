#!/usr/bin/env python3
"""
AI Data Science Orchestrator Server - A2A Compatible
Universal AI-driven orchestration using dynamic agent discovery and LLM reasoning
Based on A2A protocol research and best practices
"""

import logging
import uvicorn
import os
import sys
from dotenv import load_dotenv
import json
import asyncio
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.client import A2ACardResolver, A2AClient

# Core imports
from core.llm_factory import create_llm_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiscoveredAgent:
    """Represents a dynamically discovered A2A agent."""
    name: str
    url: str
    description: str
    skills: List[Dict[str, Any]]
    capabilities: Dict[str, Any]
    agent_card: AgentCard

@dataclass
class OrchestrationPlan:
    """Represents an AI-generated orchestration plan."""
    objective: str
    reasoning: str
    steps: List[Dict[str, Any]]
    selected_agents: List[DiscoveredAgent]

class UniversalAgentDiscovery:
    """Dynamic A2A agent discovery and capability mapping."""
    
    def __init__(self):
        """Initialize agent discovery system."""
        # Default agent endpoints for discovery (excluding pandas_data_analyst)
        self.discovery_endpoints = [
            "http://localhost:8203",  # EDA Tools
            "http://localhost:8202",  # Data Visualization
            "http://localhost:8205",  # Data Cleaning
            "http://localhost:8204",  # Feature Engineering
            "http://localhost:8000",  # Data Loader
        ]
        self.discovered_agents: Dict[str, DiscoveredAgent] = {}
        
    async def discover_available_agents(self) -> Dict[str, DiscoveredAgent]:
        """Dynamically discover available A2A agents via their Agent Cards."""
        logger.info("🔍 Starting dynamic agent discovery...")
        discovered = {}
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for endpoint in self.discovery_endpoints:
                try:
                    # Get agent card using A2A standard
                    resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
                    agent_card = await resolver.get_agent_card()
                    
                    if agent_card:
                        agent = DiscoveredAgent(
                            name=agent_card.name,
                            url=endpoint,
                            description=agent_card.description,
                            skills=[skill.model_dump() for skill in agent_card.skills],
                            capabilities=agent_card.capabilities.model_dump() if agent_card.capabilities else {},
                            agent_card=agent_card
                        )
                        discovered[agent.name] = agent
                        logger.info(f"✅ Discovered: {agent.name} at {endpoint}")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Could not discover agent at {endpoint}: {e}")
                    
        self.discovered_agents = discovered
        logger.info(f"🎯 Discovery complete: {len(discovered)} agents found")
        return discovered

class LLMOrchestrationEngine:
    """AI-driven orchestration engine using LLM reasoning."""
    
    def __init__(self):
        """Initialize LLM-based orchestration engine."""
        self.llm = create_llm_instance()
        
    async def create_orchestration_plan(
        self, 
        user_request: str, 
        available_agents: Dict[str, DiscoveredAgent],
        data_context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationPlan:
        """Generate orchestration plan using LLM reasoning."""
        
        # Prepare agent capabilities summary for LLM
        agents_summary = self._prepare_agents_summary(available_agents)
        data_summary = self._prepare_data_summary(data_context) if data_context else "No specific data context provided."
        
        # Universal orchestration prompt - domain agnostic
        orchestration_prompt = f"""You are a Universal AI Orchestrator for multi-agent systems following the A2A (Agent-to-Agent) protocol.

Your role is to analyze user requests and create optimal multi-agent collaboration plans using available specialized agents.

**USER REQUEST:**
{user_request}

**DATA CONTEXT:**
{data_summary}

**AVAILABLE AGENTS:**
{agents_summary}

**ORCHESTRATION PRINCIPLES:**
1. Each agent has opaque execution - you cannot see their internal implementation
2. Select agents based on their declared skills and examples
3. Consider emergent capabilities agents might have beyond declared skills
4. Create logical task decomposition that maximizes agent strengths
5. Ensure proper task sequencing for dependencies
6. Be universal - don't assume specific domains or datasets

**OUTPUT FORMAT (JSON):**
{{
  "objective": "Clear statement of what we're trying to achieve",
  "reasoning": "Detailed explanation of your orchestration strategy and agent selection rationale",
  "steps": [
    {{
      "step_number": 1,
      "agent_name": "Exact agent name from available agents",
      "task_description": "Specific task for this agent to perform",
      "reasoning": "Why this agent was selected for this specific task",
      "dependencies": ["List of previous steps this depends on"],
      "expected_output": "What type of result this step should produce"
    }}
  ]
}}

Analyze the request and create the optimal orchestration plan."""

        try:
            response = await self.llm.ainvoke(orchestration_prompt)
            plan_json = self._extract_json_from_response(response.content)
            
            # Validate and create orchestration plan
            return self._create_validated_plan(plan_json, available_agents, user_request)
            
        except Exception as e:
            logger.error(f"❌ Orchestration planning failed: {e}")
            # Fallback to simple analysis plan
            return self._create_fallback_plan(user_request, available_agents)
            
    def _prepare_agents_summary(self, agents: Dict[str, DiscoveredAgent]) -> str:
        """Prepare a summary of available agents for LLM analysis."""
        summary = []
        for agent_name, agent in agents.items():
            skills_text = ""
            for skill in agent.skills:
                examples = ", ".join(skill.get("examples", []))
                skills_text += f"  - {skill['name']}: {skill['description']}\n"
                if examples:
                    skills_text += f"    Examples: {examples}\n"
                    
            summary.append(f"""
**{agent_name}**
URL: {agent.url}
Description: {agent.description}
Skills:
{skills_text}""")
            
        return "\n".join(summary)
        
    def _prepare_data_summary(self, data_context: Dict[str, Any]) -> str:
        """Prepare data context summary for LLM analysis."""
        if not data_context:
            return "No specific data context provided."
            
        parts = []
        if "data_id" in data_context:
            parts.append(f"Data ID: {data_context['data_id']}")
        if "shape" in data_context:
            parts.append(f"Shape: {data_context['shape']}")
        if "columns" in data_context:
            parts.append(f"Columns: {data_context['columns']}")
        if "data_type" in data_context:
            parts.append(f"Type: {data_context['data_type']}")
            
        return ", ".join(parts) if parts else "Basic data context available."
        
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling various formats."""
        try:
            # Try direct JSON parsing
            return json.loads(response_text)
        except (json.JSONDecodeError, ValueError):
            # Try to find JSON within text
            start_markers = ["{", "```json", "```"]
            end_markers = ["}", "```"]
            
            for start_marker in start_markers:
                start_idx = response_text.find(start_marker)
                if start_idx != -1:
                    if start_marker == "{":
                        # Find matching closing brace
                        brace_count = 0
                        for i, char in enumerate(response_text[start_idx:], start_idx):
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    try:
                                        json_text = response_text[start_idx:i+1]
                                        return json.loads(json_text)
                                    except (json.JSONDecodeError, ValueError):
                                        continue
                    else:
                        # Find end marker
                        for end_marker in end_markers:
                            end_idx = response_text.find(end_marker, start_idx + len(start_marker))
                            if end_idx != -1:
                                try:
                                    json_text = response_text[start_idx + len(start_marker):end_idx]
                                    return json.loads(json_text)
                                except (json.JSONDecodeError, ValueError):
                                    continue
                                
            # If all else fails, create a structured response from the text
            return {
                "objective": "Analyze and process the user request",
                "reasoning": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "steps": [
                    {
                        "step_number": 1,
                        "agent_name": "EDA Tools Agent",
                        "task_description": "Perform comprehensive analysis based on user request",
                        "reasoning": "Selected based on available capabilities",
                        "dependencies": [],
                        "expected_output": "Analysis results and insights"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"❌ JSON extraction failed: {e}")
            raise
            
    def _create_validated_plan(
        self, 
        plan_json: Dict[str, Any], 
        available_agents: Dict[str, DiscoveredAgent],
        user_request: str
    ) -> OrchestrationPlan:
        """Create and validate orchestration plan from LLM response."""
        
        # Validate required fields
        objective = plan_json.get("objective", "Process user request")
        reasoning = plan_json.get("reasoning", "AI-generated orchestration plan")
        steps = plan_json.get("steps", [])
        
        # Validate agents in steps exist
        selected_agents = []
        validated_steps = []
        
        for step in steps:
            agent_name = step.get("agent_name", "")
            if agent_name in available_agents:
                selected_agents.append(available_agents[agent_name])
                validated_steps.append(step)
            else:
                # Try to find closest match
                for available_name in available_agents.keys():
                    if agent_name.lower() in available_name.lower() or available_name.lower() in agent_name.lower():
                        step["agent_name"] = available_name
                        selected_agents.append(available_agents[available_name])
                        validated_steps.append(step)
                        break
                        
        if not validated_steps:
            # Fallback: use first available agent
            first_agent = list(available_agents.values())[0]
            validated_steps = [{
                "step_number": 1,
                "agent_name": first_agent.name,
                "task_description": user_request,
                "reasoning": "Fallback selection based on available agents",
                "dependencies": [],
                "expected_output": "Analysis results"
            }]
            selected_agents = [first_agent]
            
        return OrchestrationPlan(
            objective=objective,
            reasoning=reasoning,
            steps=validated_steps,
            selected_agents=selected_agents
        )
        
    def _create_fallback_plan(
        self, 
        user_request: str, 
        available_agents: Dict[str, DiscoveredAgent]
    ) -> OrchestrationPlan:
        """Create a simple fallback plan when LLM reasoning fails."""
        
        if not available_agents:
            raise ValueError("No agents available for orchestration")
            
        # Simple heuristic: use first available agent
        first_agent = list(available_agents.values())[0]
        
        return OrchestrationPlan(
            objective="Process user request with available agents",
            reasoning="Fallback orchestration due to planning complexity",
            steps=[{
                "step_number": 1,
                "agent_name": first_agent.name,
                "task_description": user_request,
                "reasoning": "Selected as primary available agent",
                "dependencies": [],
                "expected_output": "Analysis results and insights"
            }],
            selected_agents=[first_agent]
        )

class UniversalOrchestratorExecutor(AgentExecutor):
    """Universal A2A Orchestrator with AI-driven agent discovery and coordination."""
    
    def __init__(self):
        """Initialize universal orchestrator."""
        self.agent_discovery = UniversalAgentDiscovery()
        self.orchestration_engine = LLMOrchestrationEngine()
        self.discovered_agents: Dict[str, DiscoveredAgent] = {}
        
    async def execute(self, context: RequestContext) -> None:
        """Execute universal orchestration with dynamic agent discovery."""
        event_queue: EventQueue = context.deps.event_queue
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user request and context
            user_request, data_context = self._extract_request_context(context)
            
            # Dynamic agent discovery
            logger.info("🔍 Starting dynamic agent discovery...")
            available_agents = await self.agent_discovery.discover_available_agents()
            
            if not available_agents:
                raise ValueError("❌ No A2A agents discovered. Please ensure agent servers are running.")
                
            # AI-driven orchestration planning  
            logger.info("🧠 Generating AI orchestration plan...")
            orchestration_plan = await self.orchestration_engine.create_orchestration_plan(
                user_request, available_agents, data_context
            )
            
            # Generate response
            response = self._format_orchestration_response(
                orchestration_plan, user_request, data_context
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(response)
            )
            
        except Exception as e:
            logger.error(f"❌ Universal orchestration failed: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Orchestration failed: {str(e)}")
            )
            raise RuntimeError(f"Universal orchestration failed: {str(e)}") from e
            
    def _extract_request_context(self, context: RequestContext) -> tuple[str, Optional[Dict[str, Any]]]:
        """Extract user request and data context from A2A message."""
        user_request = ""
        data_context = None
        
        if context.message and context.message.parts:
            for part in context.message.parts:
                if part.kind == "text":
                    content = part.text
                    user_request += content + " "
                    
                    # Extract data context if present
                    if "data_info:" in content:
                        try:
                            data_start = content.find("data_info:") + len("data_info:")
                            data_json = content[data_start:].strip()
                            data_context = json.loads(data_json)
                        except Exception as e:
                            logger.warning(f"⚠️  Could not parse data context: {e}")
                            
        return user_request.strip(), data_context
        
    def _format_orchestration_response(
        self, 
        plan: OrchestrationPlan, 
        user_request: str,
        data_context: Optional[Dict[str, Any]]
    ) -> str:
        """Format orchestration plan as comprehensive response."""
        
        response_parts = []
        
        # Header
        response_parts.append("# 🎯 Universal AI Orchestration Plan\n")
        response_parts.append(f"**User Request**: {user_request}\n\n")
        
        # Objective and Reasoning
        response_parts.append("## 🎯 Objective\n")
        response_parts.append(f"{plan.objective}\n\n")
        
        response_parts.append("## 🧠 AI Reasoning\n")
        response_parts.append(f"{plan.reasoning}\n\n")
        
        # Data Context
        if data_context:
            response_parts.append("## 📊 Data Context\n")
            for key, value in data_context.items():
                response_parts.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")
            response_parts.append("\n")
            
        # Discovered Agents
        response_parts.append("## 🤖 Discovered Agents\n")
        for agent in plan.selected_agents:
            response_parts.append(f"### {agent.name}\n")
            response_parts.append(f"- **Endpoint**: {agent.url}\n")
            response_parts.append(f"- **Description**: {agent.description}\n")
            response_parts.append(f"- **Skills**: {len(agent.skills)} specialized capabilities\n\n")
            
        # Execution Plan
        response_parts.append("## 🚀 Execution Plan\n")
        for step in plan.steps:
            step_num = step.get("step_number", 1)
            agent_name = step.get("agent_name", "Unknown Agent")
            task_desc = step.get("task_description", "")
            reasoning = step.get("reasoning", "")
            expected_output = step.get("expected_output", "Analysis results")
            
            response_parts.append(f"### Step {step_num}: {agent_name}\n")
            response_parts.append(f"**Task**: {task_desc}\n\n")
            response_parts.append(f"**Reasoning**: {reasoning}\n\n")
            response_parts.append(f"**Expected Output**: {expected_output}\n\n")
            response_parts.append("---\n\n")
            
        # Footer
        response_parts.append("## ✨ Next Steps\n")
        response_parts.append("This AI-generated orchestration plan leverages the A2A protocol for seamless agent collaboration. ")
        response_parts.append("Each agent will execute its specialized tasks independently while contributing to the overall objective.\n\n")
        response_parts.append("**Orchestration Philosophy**: Universal, AI-driven, protocol-compliant multi-agent coordination.")
        
        return "".join(response_parts)
        
    async def cancel(self, context: RequestContext) -> None:
        """Cancel orchestration execution."""
        logger.info("🛑 Cancelling universal orchestration...")

def main():
    """Main function to start the universal orchestrator server."""
    skill = AgentSkill(
        id="universal_orchestration",
        name="Universal AI Orchestration",
        description="AI-driven orchestration of multi-agent workflows with dynamic agent discovery and intelligent task decomposition",
        tags=["orchestration", "ai-driven", "universal", "multi-agent", "a2a-protocol"],
        examples=[
            "analyze my dataset comprehensively", 
            "coordinate agents for data analysis", 
            "create intelligent workflow for my data",
            "orchestrate specialized agents for insights",
            "plan multi-step analysis strategy"
        ]
    )

    agent_card = AgentCard(
        name="Universal AI Orchestrator",
        description="An AI-driven orchestrator that dynamically discovers A2A agents and creates intelligent multi-agent collaboration plans using LLM reasoning.",
        url="http://localhost:8100/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=UniversalOrchestratorExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("🎯 Starting Universal AI Orchestrator Server")
    print("🌐 Server starting on http://localhost:8100")
    print("📋 Agent card: http://localhost:8100/.well-known/agent.json")
    print("🧠 Features: AI-driven agent discovery & universal orchestration")

    uvicorn.run(server.build(), host="0.0.0.0", port=8100, log_level="info")

if __name__ == "__main__":
    main()
