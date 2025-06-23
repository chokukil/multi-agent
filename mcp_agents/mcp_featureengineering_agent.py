from fastapi import APIRouter, FastAPI
from core.agents.base_a2a_agent import BaseA2AAgent
import logging
from core.data_manager import DataManager
from core.schemas.messages import A2ARequest, A2AResponse, DataFrameContent, StatusContent
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from a2a.sdk.types import AgentCard, Skill

class FeatureEngineeringAgent(BaseA2AAgent):
    """
    An agent for creating new features in a dataset.
    """

    def get_agent_card(self) -> "AgentCard":
        return AgentCard(
            name="FeatureEngineeringAgent",
            description="Creates new features from existing ones.",
            version="1.0.0",
            skills=[
                Skill(
                    name="create_polynomial_features",
                    description="Creates polynomial and interaction features.",
                    parameters={
                        "input_df_id": "string",
                        "output_df_id": "string",
                        "columns": "array[string]",
                        "degree": "integer (default: 2)"
                    },
                    returns="object (A summary of the new dataframe)",
                ),
            ],
        )

    def register_skills(self):
        pass

    async def process_request(self, request: A2ARequest) -> A2AResponse:
        if request.action == "create_polynomial_features":
            return await self.create_polynomial_features(request)
        else:
            return A2AResponse(
                status="error",
                message=f"Unknown action: {request.action}",
                contents=[StatusContent(data={"status": "error", "message": f"Unknown action: {request.action}"})]
            )

    async def create_polynomial_features(self, request: A2ARequest) -> A2AResponse:
        params = request.contents[0].data
        input_df_id = params.get("input_df_id")
        output_df_id = params.get("output_df_id")
        columns = params.get("columns")
        degree = params.get("degree", 2)

        if not all([input_df_id, output_df_id, columns]):
            return A2AResponse(status="error", message="Missing required parameters: 'input_df_id', 'output_df_id', 'columns'.")

        logging.info(f"FeatureEngineeringAgent: Creating polynomial features for {input_df_id}")

        try:
            data_manager = DataManager()
            df = data_manager.get_dataframe(input_df_id)
            if df is None:
                return A2AResponse(status="error", message=f"DataFrame with ID '{input_df_id}' not found.")

            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[columns])
            
            poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(columns))
            
            # Drop original columns and concatenate new ones
            new_df = df.drop(columns=columns).join(poly_df)

            source_name = f"poly_features_{degree}_{input_df_id}"
            data_manager.add_dataframe(output_df_id, new_df, source=source_name)

            return A2AResponse(
                status="success",
                message=f"Polynomial features created in DataFrame '{output_df_id}'.",
                contents=[DataFrameContent(data={"df_id": output_df_id, "row_count": len(new_df), "columns": list(new_df.columns)})]
            )
        except Exception as e:
            logging.error(f"Error creating polynomial features for {input_df_id}: {e}", exc_info=True)
            return A2AResponse(status="error", message=f"An error occurred: {str(e)}")


app = FastAPI(title="FeatureEngineeringAgent")
router = APIRouter()
agent_instance = FeatureEngineeringAgent(config_path="mcp-configs/featureengineering_agent.json", api_router=router)

@router.post("/process", response_model=A2AResponse)
async def process(request: A2ARequest) -> A2AResponse:
    return await agent_instance.process_request(request)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=agent_instance.host, port=agent_instance.port) 