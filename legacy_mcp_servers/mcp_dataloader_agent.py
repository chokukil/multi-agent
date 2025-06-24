import pandas as pd
import os
import logging
from typing import List

from fastapi import APIRouter, HTTPException
from core.agents.base_a2a_agent import BaseA2AAgent
from core.data_manager import data_manager
from core.schemas.messages import A2ARequest, A2AResponse, DataFrameContent, ParamsContent, AnyContent

# Create a router for the custom endpoints of this agent
api_router = APIRouter()

@api_router.post("/process", response_model=A2AResponse)
async def process_request(request: A2ARequest):
    """
    Main processing endpoint for the DataLoaderAgent.
    It currently only handles the 'load_data' action.
    """
    if request.action == "load_data":
        try:
            # Extract parameters from the request
            params_content = next((c for c in request.contents if isinstance(c, ParamsContent)), None)
            if not params_content:
                raise HTTPException(status_code=400, detail="Parameters not provided for load_data action.")

            file_path = params_content.data.get("file_path")
            output_df_id = params_content.data.get("output_df_id")

            if not file_path or not output_df_id:
                raise HTTPException(status_code=400, detail="Missing 'file_path' or 'output_df_id' in parameters.")

            logging.info(f"Loading data from '{file_path}' with id '{output_df_id}'")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No such file or directory: '{file_path}'")
            
            df = pd.read_csv(file_path)
            data_manager.add_dataframe(data_id=output_df_id, data=df)
            
            response_content = DataFrameContent(data={"df_id": output_df_id})
            return A2AResponse(
                status="success",
                message=f"DataFrame '{output_df_id}' loaded successfully.",
                contents=[response_content]
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logging.error(f"Error during load_data action: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Action '{request.action}' not supported.")


class DataLoaderAgent(BaseA2AAgent):
    """
    An agent responsible for loading data from various file formats into the DataManager.
    Custom endpoints are handled via the included api_router.
    """
    def register_skills(self):
        """No A2A skills are registered directly. Using APIRouter instead."""
        pass

# --- Server Setup ---
try:
    dataloader_agent = DataLoaderAgent(
        config_path="mcp-configs/dataloader_agent.json",
        api_router=api_router
    )
    app = dataloader_agent.app
except Exception as e:
    logging.error(f"Failed to initialize DataLoaderAgent: {e}", exc_info=True)
    app = None

if __name__ == "__main__":
    if dataloader_agent:
        dataloader_agent.start()
    else:
        logging.error("Could not start DataLoaderAgent due to initialization failure.") 