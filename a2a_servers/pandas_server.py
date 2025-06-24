import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import pandas as pd
import traceback
import atexit

from python_a2a import A2AServer, skill, agent, run_server, A2AClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from core_logic.multiagents.pandas_data_analyst import PandasDataAnalyst
from core_logic.agents.data_wrangling_agent import DataWranglingAgent
from core_logic.agents.data_visualization_agent import DataVisualizationAgent
from config import AGENT_SERVERS, REGISTRY_URL, OPENAI_API_KEY

# Agent specific configuration
AGENT_CONFIG = AGENT_SERVERS["pandas_analyst"]
AGENT_NAME = "Pandas Data Analyst"

# Initialize components
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
checkpointer = MemorySaver()

# This is the core logic agent from the original codebase
pandas_analyst_logic = PandasDataAnalyst(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(model=llm, checkpointer=checkpointer, human_in_the_loop=True),
    data_visualization_agent=DataVisualizationAgent(model=llm, checkpointer=checkpointer),
    checkpointer=checkpointer
)

# A2A Client to communicate with the Registry
registry_client = A2AClient(REGISTRY_URL)

@agent(
    name=AGENT_NAME,
    description="Automated data analysis and visualization using Pandas. Streams the process step-by-step.",
    version="1.0.0"
)
class PandasAnalystA2AServer(A2AServer):

    async def _log_event(self, task_id, event_type, content=""):
        try:
            await registry_client.ask(
                "log_event",
                agent_name=AGENT_NAME,
                task_id=task_id,
                event_type=event_type,
                content=str(content)
            )
        except Exception as e:
            print(f"Failed to log event: {e}")

    def format_event_for_client(self, event: dict, task_id: str):
        """Formats a LangGraph event into a standardized JSON chunk for the client."""
        event_name = event['event']
        run_id = str(event['run_id'])
        node_name = event.get('name', 'graph')
        
        chunk = {
            "task_id": task_id,
            "event": event_name,
            "node_name": node_name,
            "run_id": run_id,
            "status": "running",
            "output": event['data'].get('output') or event['data'].get('chunk')
        }

        # Check for human_in_the_loop trigger
        if "human_review" in node_name and event_name == "on_chain_stream":
            chunk["status"] = "human_input_required"
            # The 'output' here contains the code to be reviewed
            chunk["output"] = event['data'].get('chunk', {}).get('messages', [{}])[0].content
            
        return chunk

    @skill(
        name="run_pandas_analysis_stream",
        description="Streams the process of pandas analysis step-by-step."
    )
    async def run_pandas_analysis_stream(self, user_instructions: str, data_raw_json: str, session_id: str):
        await self._log_event(session_id, "skill_start", f"User instructions: {user_instructions}")

        try:
            data_raw = pd.read_json(data_raw_json, orient='split')
            config = {"configurable": {"thread_id": session_id}}
            
            async for event in pandas_analyst_logic.astream_events(
                {"user_instructions": user_instructions, "data_raw": data_raw},
                config=config,
                version="v1"
            ):
                chunk = self.format_event_for_client(event, session_id)
                await self._log_event(session_id, "event_stream", json.dumps(chunk))
                yield json.dumps(chunk)
            
            # After stream, get final result
            final_response = pandas_analyst_logic.response
            final_chunk = {
                "task_id": session_id,
                "event": "final_result",
                "status": "completed",
                "output": {
                    "wrangled_data": final_response.get_data_wrangled().to_json(orient='split'),
                    "plot": final_response.get_plotly_graph().to_json() if final_response.get_plotly_graph() else None,
                    "wrangler_code": final_response.get_data_wrangler_function(),
                    "visualization_code": final_response.get_data_visualization_function(),
                    "summary": final_response.get_workflow_summary(markdown=True),
                }
            }
            await self._log_event(session_id, "skill_success", "Analysis completed successfully.")
            yield json.dumps(final_chunk)

        except Exception as e:
            tb = traceback.format_exc()
            await self._log_event(session_id, "skill_error", tb)
            yield json.dumps({"task_id": session_id, "event": "error", "status": "error", "output": str(e)})


async def register_agent():
    """Registers this agent with the registry server."""
    my_card = {
        "name": AGENT_NAME,
        "description": "Automated data analysis and visualization using Pandas.",
        "url": AGENT_CONFIG["url"],
        "version": "1.0.0",
        "skills": ["run_pandas_analysis_stream"]
    }
    try:
        await registry_client.ask("register", agent_name=AGENT_NAME, url=AGENT_CONFIG["url"], agent_card=my_card)
        print(f"Agent '{AGENT_NAME}' registered successfully.")
    except Exception as e:
        print(f"Failed to register agent '{AGENT_NAME}': {e}")

async def unregister_agent():
    """Unregisters this agent from the registry server."""
    try:
        await registry_client.ask("unregister", agent_name=AGENT_NAME)
        print(f"Agent '{AGENT_NAME}' unregistered successfully.")
    except Exception as e:
        print(f"Failed to unregister agent '{AGENT_NAME}': {e}")

async def send_heartbeat():
    """Sends a heartbeat to the registry server periodically."""
    while True:
        try:
            await registry_client.ask("heartbeat", agent_name=AGENT_NAME)
        except Exception as e:
            print(f"Failed to send heartbeat for '{AGENT_NAME}': {e}")
        await asyncio.sleep(15)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    
    # Register on startup
    loop.run_until_complete(register_agent())
    
    # Schedule heartbeat
    heartbeat_task = loop.create_task(send_heartbeat())
    
    # Register unregister on exit
    atexit.register(lambda: loop.run_until_complete(unregister_agent()))

    server = PandasAnalystA2AServer(google_a2a_compatible=True)
    print(f"Starting Pandas Analyst Server on {AGENT_CONFIG['url']}")
    run_server(server, host=AGENT_CONFIG['host'], port=AGENT_CONFIG['port'], stream_type='sse', debug=False) 