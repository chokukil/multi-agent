from core.agents.base_a2a_agent import BaseA2AAgent, logger
from core.data_manager import DataManager
from a2a.sdk.types import AgentCard, Skill, Task, TaskResult, Media
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

class H2OModelingAgent(BaseA2AAgent):
    """
    An agent that uses H2O's AutoML to build and evaluate machine learning models.
    """

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        h2o.init() # Initialize H2O cluster

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="H2OModelingAgent",
            description="Uses H2O AutoML to find the best models for a dataset.",
            version="1.0.0",
            skills=[
                Skill(
                    name="run_automl",
                    description="Runs H2O AutoML for classification or regression.",
                    parameters={
                        "data_id": "string",
                        "target_column": "string",
                        "max_runtime_secs": "integer (optional, default 60)"
                    },
                    returns="string (leaderboard_data_id)",
                ),
            ],
        )

    def _register_skills(self):
        self.a2a_server.skill("run_automl")(self.run_automl)

    async def run_automl(self, task: Task) -> TaskResult:
        data_id = task.parameters.get("data_id")
        target_column = task.parameters.get("target_column")
        max_runtime_secs = task.parameters.get("max_runtime_secs", 60)

        if not all([data_id, target_column]):
            return TaskResult(status="error", message="Missing 'data_id' or 'target_column' parameter.")

        logger.info(f"H2OModelingAgent: Starting AutoML for {data_id}, target='{target_column}'")
        try:
            dm = DataManager()
            df = dm.get_dataframe(data_id)
            if df is None:
                return TaskResult(status="error", message=f"Data ID '{data_id}' not found.")
            
            h2o_df = h2o.H2OFrame(df)
            
            # For classification, the target column must be a factor
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20: # Heuristic
                 h2o_df[target_column] = h2o_df[target_column].asfactor()

            x = h2o_df.columns
            y = target_column
            x.remove(y)

            aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1)
            aml.train(x=x, y=y, training_frame=h2o_df)

            leaderboard = aml.leaderboard.as_data_frame()
            
            source = f"h2o_leaderboard_{data_id}"
            lb_data_id = dm.add_dataframe(leaderboard, source=source)
            
            if lb_data_id:
                return TaskResult(status="success", data=[Media(value=lb_data_id)])
            else:
                return TaskResult(status="error", message="Failed to save leaderboard to DataManager.")
        except Exception as e:
            logger.error(f"Error during AutoML run: {e}", exc_info=True)
            return TaskResult(status="error", message=f"An error occurred during AutoML: {str(e)}")

if __name__ == "__main__":
    agent = H2OModelingAgent(host="127.0.0.1", port=8008)
    agent.start() 