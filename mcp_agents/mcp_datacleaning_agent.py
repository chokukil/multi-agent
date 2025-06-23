from fastapi import APIRouter, FastAPI
from core.agents.base_a2a_agent import BaseA2AAgent
import logging
from core.data_manager import DataManager
from core.schemas.messages import A2ARequest, A2AResponse, DataFrameContent, StatusContent
import pandas as pd
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from a2a.sdk.types import AgentCard, Skill

class DataCleaningAgent(BaseA2AAgent):
    """
    An agent expert in data cleaning tasks like handling missing values.
    """

    def get_agent_card(self) -> "AgentCard":
        return AgentCard(
            name="DataCleaningAgent",
            description="Cleans data by handling missing values.",
            version="1.0.0",
            skills=[
                Skill(
                    name="handle_missing_values",
                    description="Handles missing values in a DataFrame using a specified strategy.",
                    parameters={
                        "input_df_id": "string",
                        "output_df_id": "string",
                        "strategy": "string (e.g., 'mean', 'median', 'mode', 'ffill', 'bfill', or a constant value)",
                        "columns": "array[string] (optional, specific columns to apply the strategy)"
                    },
                    returns="object (A summary of the cleaned dataframe)",
                ),
            ],
        )

    def register_skills(self):
        """No skills to register with a2a, handled by FastAPI endpoint."""
        pass

    async def process_request(self, request: A2ARequest) -> A2AResponse:
        if request.action == "handle_missing_values":
            return await self.handle_missing_values(request)
        else:
            return A2AResponse(
                status="error",
                message=f"Unknown action: {request.action}",
                contents=[StatusContent(data={"status": "error", "message": f"Unknown action: {request.action}"})]
            )

    async def handle_missing_values(self, request: A2ARequest) -> A2AResponse:
        params = request.contents[0].data
        input_df_id = params.get("input_df_id")
        output_df_id = params.get("output_df_id")
        strategy = params.get("strategy", "mean")
        columns: Optional[List[str]] = params.get("columns")

        if not input_df_id or not output_df_id:
            return A2AResponse(status="error", message="Missing 'input_df_id' or 'output_df_id' parameter.")
        
        logging.info(f"DataCleaningAgent: Handling missing values for {input_df_id} -> {output_df_id}")

        try:
            data_manager = DataManager()
            df = data_manager.get_dataframe(input_df_id)

            if df is None:
                return A2AResponse(status="error", message=f"DataFrame with ID '{input_df_id}' not found.")
            
            cleaned_df = df.copy()
            target_cols = columns if columns else cleaned_df.columns

            for col in target_cols:
                if col not in cleaned_df.columns:
                    logging.warning(f"Column '{col}' not found in DataFrame {input_df_id}. Skipping.")
                    continue
                
                if strategy in ['mean', 'median']:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        fill_value = getattr(cleaned_df[col], strategy)()
                        cleaned_df[col].fillna(fill_value, inplace=True)
                elif strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                    cleaned_df[col].fillna(fill_value, inplace=True)
                elif strategy in ['ffill', 'bfill']:
                    cleaned_df[col].fillna(method=strategy, inplace=True)
                else: 
                    try:
                        dtype = cleaned_df[col].dtype
                        fill_value = pd.Series([strategy]).astype(dtype).iloc[0]
                    except (ValueError, TypeError):
                        fill_value = strategy
                    cleaned_df[col].fillna(fill_value, inplace=True)

            source_name = f"cleaned_{strategy}_{input_df_id}"
            data_manager.add_dataframe(output_df_id, cleaned_df, source=source_name)

            return A2AResponse(
                status="success",
                message=f"DataFrame '{output_df_id}' created with cleaned data.",
                contents=[DataFrameContent(data={"df_id": output_df_id, "row_count": len(cleaned_df), "columns": list(cleaned_df.columns)})]
            )
        except Exception as e:
            logging.error(f"Error handling missing values for {input_df_id}: {e}", exc_info=True)
            return A2AResponse(status="error", message=f"An error occurred: {str(e)}")

app = FastAPI(title="DataCleaningAgent")
router = APIRouter()
agent_instance = DataCleaningAgent(config_path="mcp-configs/datacleaning_agent.json", api_router=router)

@router.post("/process", response_model=A2AResponse)
async def process(request: A2ARequest) -> A2AResponse:
    return await agent_instance.process_request(request)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=agent_instance.host, port=agent_instance.port) 