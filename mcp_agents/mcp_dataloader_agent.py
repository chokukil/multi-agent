from fastapi import APIRouter, FastAPI
from core.agents.base_a2a_agent import BaseA2AAgent
import logging
from core.data_manager import DataManager
from core.schemas.messages import A2ARequest, A2AResponse, DataFrameContent, StatusContent
import pandas as pd
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from a2a.sdk.types import AgentCard, Skill

class DataLoaderAgent(BaseA2AAgent):
    """
    An agent responsible for loading data from various file formats into the DataManager.
    """

    def get_agent_card(self) -> "AgentCard":
        return AgentCard(
            name="DataLoaderAgent",
            description="Loads data from various file formats (e.g., CSV, Excel) into the system.",
            version="1.0.0",
            skills=[
                Skill(
                    name="load_data",
                    description="Loads a data file and stores it in the data manager.",
                    parameters={
                        "file_path": "string (The local path to the data file.)",
                        "output_df_id": "string (The ID to assign to the loaded dataframe.)"
                    },
                    returns="object (A summary of the loaded dataframe)",
                ),
            ],
        )

    def register_skills(self):
        """No skills to register with a2a, handled by FastAPI endpoint."""
        pass

    async def process_request(self, request: A2ARequest) -> A2AResponse:
        if request.action == "load_data":
            return await self.load_data(request)
        else:
            return A2AResponse(
                status="error",
                message=f"Unknown action: {request.action}",
                contents=[StatusContent(data={"status": "error", "message": f"Unknown action: {request.action}"})]
            )

    async def load_data(self, request: A2ARequest) -> A2AResponse:
        params = request.contents[0].data
        file_path = params.get("file_path")
        output_df_id = params.get("output_df_id")

        if not file_path or not output_df_id:
            return A2AResponse(status="error", message="Missing 'file_path' or 'output_df_id' parameter.")

        logging.info(f"DataLoaderAgent: Loading data from {file_path} into {output_df_id}")
        try:
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"File not found at path: {file_path}")

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return A2AResponse(status="error", message=f"Unsupported file type for {file_path}")

            dm = DataManager()
            dm.add_dataframe(output_df_id, df, source=file_path)

            return A2AResponse(
                status="success",
                message=f"DataFrame '{output_df_id}' loaded successfully from {file_path}. Shape: {df.shape}",
                contents=[DataFrameContent(data={"df_id": output_df_id, "row_count": len(df), "columns": list(df.columns)})]
            )
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            return A2AResponse(status="error", message=str(e))


app = FastAPI(title="DataLoaderAgent")
router = APIRouter()
agent_instance = DataLoaderAgent(
    config_path="mcp-configs/dataloader_agent.json",
    api_router=router
)

@router.post("/process", response_model=A2AResponse)
async def process(request: A2ARequest) -> A2AResponse:
    return await agent_instance.process_request(request)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=agent_instance.host, port=agent_instance.port) 