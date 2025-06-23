from fastapi import APIRouter, FastAPI
from core.agents.base_a2a_agent import BaseA2AAgent
import logging
from core.data_manager import DataManager
from core.schemas.messages import A2ARequest, A2AResponse, MediaContent, StatusContent
import pandas as pd
from ydata_profiling import ProfileReport
import os
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from a2a.sdk.types import AgentCard, Skill

class EDAAgent(BaseA2AAgent):
    """
    An agent that performs Exploratory Data Analysis (EDA) on a dataframe.
    """

    def get_agent_card(self) -> "AgentCard":
        return AgentCard(
            name="EDAAgent",
            description="Performs Exploratory Data Analysis.",
            version="1.0.0",
            skills=[
                Skill(
                    name="get_descriptive_statistics",
                    description="Returns key descriptive statistics for a dataframe as JSON.",
                    parameters={"data_id": "string"},
                    returns="object",
                ),
                Skill(
                    name="generate_profile_report",
                    description="Generates a full HTML EDA report and returns its file path.",
                    parameters={"data_id": "string", "title": "string"},
                    returns="string (file_path)",
                ),
            ],
        )

    def register_skills(self):
        """No skills to register with a2a, handled by FastAPI endpoint."""
        pass

    async def process_request(self, request: A2ARequest) -> A2AResponse:
        action_map = {
            "get_descriptive_statistics": self.get_descriptive_statistics,
            "generate_profile_report": self.generate_profile_report,
        }
        action_func = action_map.get(request.action)

        if action_func:
            return await action_func(request)
        else:
            return A2AResponse(
                status="error",
                message=f"Unknown action: {request.action}",
                contents=[StatusContent(data={"status": "error", "message": f"Unknown action: {request.action}"})]
            )

    async def get_descriptive_statistics(self, request: A2ARequest) -> A2AResponse:
        params = request.contents[0].data
        data_id = params.get("data_id")
        if not data_id:
            return A2AResponse(status="error", message="Missing 'data_id' parameter.")

        logging.info(f"EDAAgent: Calculating descriptive statistics for {data_id}")
        try:
            dm = DataManager()
            df = dm.get_dataframe(data_id)
            if df is None: return A2AResponse(status="error", message=f"Data ID '{data_id}' not found.")

            stats = df.describe(include='all').to_json(orient='columns')
            return A2AResponse(status="success", message="Statistics calculated.", contents=[MediaContent(data=json.loads(stats))])
        except Exception as e:
            logging.error(f"Error calculating statistics: {e}")
            return A2AResponse(status="error", message=str(e))

    async def generate_profile_report(self, request: A2ARequest) -> A2AResponse:
        params = request.contents[0].data
        data_id = params.get("data_id")
        title = params.get("title", "EDA Report")
        if not data_id:
            return A2AResponse(status="error", message="Missing 'data_id' parameter.")

        logging.info(f"EDAAgent: Generating profile report for {data_id}")
        try:
            dm = DataManager()
            df = dm.get_dataframe(data_id)
            if df is None: return A2AResponse(status="error", message=f"Data ID '{data_id}' not found.")
            
            output_dir = os.path.join("artifacts", "reports")
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, f"eda_report_{data_id}.html")

            profile = ProfileReport(df, title=title, explorative=True)
            profile.to_file(report_path)
            
            return A2AResponse(status="success", message=f"Report generated at {report_path}", contents=[MediaContent(data={"file_path": report_path})])
        except Exception as e:
            logging.error(f"Error generating profile report: {e}")
            return A2AResponse(status="error", message=str(e))

app = FastAPI(title="EDAAgent")
router = APIRouter()
agent_instance = EDAAgent(config_path="mcp-configs/eda_agent.json", api_router=router)

@router.post("/process", response_model=A2AResponse)
async def process(request: A2ARequest) -> A2AResponse:
    return await agent_instance.process_request(request)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=agent_instance.host, port=agent_instance.port) 