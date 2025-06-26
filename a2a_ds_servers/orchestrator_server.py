# A2A Data Science Servers - Orchestrator Server
# Central orchestration server for all AI Data Science Team agents
# Compatible with A2A Protocol v0.2.9

import asyncio
import json
import os
import subprocess
import signal
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# A2A SDK imports
from a2a.server.request_handlers import DefaultRequestHandler  
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.core.data_structures import AgentCard, AgentSkill, TaskState

# Local utilities
from utils.logging import setup_a2a_logger, log_agent_execution

# Setup logging
logger = setup_a2a_logger("orchestrator_server", log_file="logs/orchestrator.log")

class DataScienceOrchestrator:
    """
    Orchestrator for all AI Data Science Team A2A servers.
    Manages server lifecycle and provides unified access.
    """
    
    def __init__(self):
        self.name = "Data Science Orchestrator"
        self.servers = {}
        self.server_configs = {
            "data_loader": {
                "port": 8000,
                "module": "data_loader_server",
                "description": "Data loading and processing"
            },
            "pandas_analyst": {
                "port": 8001,
                "module": "pandas_data_analyst_server",
                "description": "Pandas data analysis with visualization"
            },
            "sql_analyst": {
                "port": 8002,
                "module": "sql_data_analyst_server",
                "description": "SQL database analysis"
            },
            "eda_tools": {
                "port": 8003,
                "module": "eda_tools_server",
                "description": "Exploratory data analysis"
            },
            "data_visualization": {
                "port": 8004,
                "module": "data_visualization_server",
                "description": "Interactive data visualizations"
            }
        }
        
        logger.info("Data Science Orchestrator initialized")

    async def start_all_servers(self) -> Dict[str, Any]:
        """Start all data science servers."""
        
        results = {"started": [], "failed": []}
        
        for server_name, config in self.server_configs.items():
            try:
                logger.info(f"Starting {server_name} server on port {config['port']}")
                
                # Start server process
                process = subprocess.Popen([
                    "python", f"{config['module']}.py"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.servers[server_name] = {
                    "process": process,
                    "config": config,
                    "started_at": datetime.now().isoformat()
                }
                
                results["started"].append({
                    "name": server_name,
                    "port": config["port"],
                    "description": config["description"]
                })
                
                # Wait a bit between starts
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to start {server_name}: {e}")
                results["failed"].append({
                    "name": server_name,
                    "error": str(e)
                })
        
        return results

    async def stop_all_servers(self) -> Dict[str, Any]:
        """Stop all data science servers."""
        
        results = {"stopped": [], "failed": []}
        
        for server_name, server_info in self.servers.items():
            try:
                process = server_info["process"]
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning(f"Force killed {server_name}")
                
                results["stopped"].append(server_name)
                logger.info(f"Stopped {server_name} server")
                
            except Exception as e:
                logger.error(f"Failed to stop {server_name}: {e}")
                results["failed"].append({
                    "name": server_name,
                    "error": str(e)
                })
        
        self.servers.clear()
        return results

    async def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers."""
        
        status = {
            "total_servers": len(self.server_configs),
            "running_servers": len(self.servers),
            "server_details": []
        }
        
        for server_name, config in self.server_configs.items():
            server_detail = {
                "name": server_name,
                "port": config["port"],
                "description": config["description"],
                "status": "running" if server_name in self.servers else "stopped"
            }
            
            if server_name in self.servers:
                server_info = self.servers[server_name]
                server_detail.update({
                    "started_at": server_info["started_at"],
                    "process_id": server_info["process"].pid
                })
            
            status["server_details"].append(server_detail)
        
        return status

    async def route_request(self, agent_type: str, user_request: str) -> Dict[str, Any]:
        """Route request to appropriate agent server."""
        
        if agent_type not in self.server_configs:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if agent_type not in self.servers:
            raise ValueError(f"Server {agent_type} is not running")
        
        config = self.server_configs[agent_type]
        
        # For now, return routing information
        # In a full implementation, this would make HTTP requests to the servers
        return {
            "routed_to": agent_type,
            "server_port": config["port"],
            "request": user_request,
            "timestamp": datetime.now().isoformat()
        }

class OrchestratorExecutor:
    """A2A Executor for the Orchestrator."""
    
    def __init__(self):
        self.orchestrator = DataScienceOrchestrator()
    
    async def execute(self, context):
        """Execute orchestrator commands with A2A TaskUpdater pattern."""
        
        event_queue = context.get_event_queue()
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_input = context.get_user_input()
            
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text="🎛️ Processing orchestrator command...")]
                )
            )
            
            # Parse command
            command = user_input.lower().strip()
            
            if "start" in command and "server" in command:
                results = await self.orchestrator.start_all_servers()
                response_text = f"🚀 **Server Startup Results**\n\n"
                response_text += f"✅ **Started**: {len(results['started'])} servers\n"
                for server in results["started"]:
                    response_text += f"  - {server['name']} (port {server['port']}): {server['description']}\n"
                
                if results["failed"]:
                    response_text += f"\n❌ **Failed**: {len(results['failed'])} servers\n"
                    for server in results["failed"]:
                        response_text += f"  - {server['name']}: {server['error']}\n"
                
            elif "stop" in command and "server" in command:
                results = await self.orchestrator.stop_all_servers()
                response_text = f"🛑 **Server Shutdown Results**\n\n"
                response_text += f"✅ **Stopped**: {', '.join(results['stopped'])}\n"
                
                if results["failed"]:
                    response_text += f"❌ **Failed to stop**: {', '.join([s['name'] for s in results['failed']])}\n"
                
            elif "status" in command:
                status = await self.orchestrator.get_server_status()
                response_text = f"📊 **Data Science Servers Status**\n\n"
                response_text += f"📈 **Running**: {status['running_servers']}/{status['total_servers']} servers\n\n"
                
                for server in status["server_details"]:
                    status_emoji = "🟢" if server["status"] == "running" else "🔴"
                    response_text += f"{status_emoji} **{server['name']}** (port {server['port']})\n"
                    response_text += f"   {server['description']}\n"
                    if server["status"] == "running":
                        response_text += f"   Started: {server.get('started_at', 'Unknown')}\n"
                    response_text += "\n"
                
            else:
                response_text = """🎛️ **Data Science Orchestrator Commands**

Available commands:
• `start servers` - Start all data science servers
• `stop servers` - Stop all data science servers  
• `status` - Check server status

**Available Servers:**
• **Data Loader** (port 8000) - Data loading and processing
• **Pandas Analyst** (port 8001) - Pandas data analysis with visualization
• **SQL Analyst** (port 8002) - SQL database analysis
• **EDA Tools** (port 8003) - Exploratory data analysis
• **Data Visualization** (port 8004) - Interactive visualizations

Example: "start servers" or "check status"
"""
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text=response_text)]
                )
            )
            
            return {"status": "success", "command": command}
            
        except Exception as e:
            error_message = f"Orchestrator command failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text=f"❌ {error_message}")]
                )
            )
            
            raise e

# Create A2A Server
def create_orchestrator_app():
    """Create the A2A application for the Orchestrator."""
    
    agent_card = AgentCard(
        name="Data Science Orchestrator",
        description="Central orchestration server managing all AI Data Science Team agents. Controls server lifecycle, monitors status, and routes requests between specialized data science agents.",
        instructions="Send orchestration commands. I can:\n"
                    "• Start and stop all data science servers\n"
                    "• Monitor server health and status\n"
                    "• Route requests to appropriate agents\n"
                    "• Provide system overview and management\n\n"
                    "Commands: 'start servers', 'stop servers', 'status'",
        skills=[
            AgentSkill(name="server_management", description="Server lifecycle management"),
            AgentSkill(name="system_monitoring", description="Server status monitoring"),
            AgentSkill(name="request_routing", description="Intelligent request routing"),
            AgentSkill(name="resource_management", description="System resource management"),
        ],
        streaming=True,
        version="1.0.0"
    )
    
    executor = OrchestratorExecutor()
    request_handler = DefaultRequestHandler(executor)
    task_store = InMemoryTaskStore()
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        request_handler=request_handler,
        task_store=task_store
    )
    
    logger.info("Data Science Orchestrator A2A Server created")
    return app

# Create the app instance
app = create_orchestrator_app()

if __name__ == "__main__":
    import uvicorn
    print("🎛️ Data Science Orchestrator A2A Server starting...")
    uvicorn.run(app, host="localhost", port=8100, log_level="info") 