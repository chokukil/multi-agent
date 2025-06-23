from core.agents.base_a2a_agent import BaseA2AAgent, logger
from core.data_manager import DataManager
from a2a.sdk.types import AgentCard, Skill, Task, TaskResult, Media
import pandas as pd
import mlflow
import os

class MLflowAgent(BaseA2AAgent):
    """
    An agent that acts as an MLOps manager, logging experiments and models to MLflow.
    """

    def __init__(self, host: str, port: int, tracking_uri: str = "http://127.0.0.1:5000"):
        super().__init__(host, port)
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="MLflowAgent",
            description="Logs experiment results and models to an MLflow Tracking Server.",
            version="1.0.0",
            skills=[
                Skill(
                    name="log_h2o_leaderboard",
                    description="Logs an H2O leaderboard to a new MLflow experiment.",
                    parameters={
                        "leaderboard_data_id": "string",
                        "experiment_name": "string"
                    },
                    returns="object (with experiment_id and run_ids)",
                ),
            ],
        )

    def _register_skills(self):
        self.a2a_server.skill("log_h2o_leaderboard")(self.log_h2o_leaderboard)

    async def log_h2o_leaderboard(self, task: Task) -> TaskResult:
        lb_data_id = task.parameters.get("leaderboard_data_id")
        exp_name = task.parameters.get("experiment_name", "h2o-automl-experiment")

        if not lb_data_id:
            return TaskResult(status="error", message="Missing 'leaderboard_data_id' parameter.")

        logger.info(f"MLflowAgent: Logging leaderboard {lb_data_id} to experiment '{exp_name}'")
        try:
            dm = DataManager()
            leaderboard_df = dm.get_dataframe(lb_data_id)
            if leaderboard_df is None:
                return TaskResult(status="error", message=f"Leaderboard data ID '{lb_data_id}' not found.")

            try:
                experiment = mlflow.create_experiment(exp_name)
                experiment_id = experiment.experiment_id
            except mlflow.exceptions.MlflowException: # Experiment already exists
                experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
            
            run_ids = []
            for i, row in leaderboard_df.iterrows():
                with mlflow.start_run(experiment_id=experiment_id, run_name=row["model_id"]) as run:
                    # Log metrics
                    metrics_to_log = {
                        "auc", "logloss", "aucpr", "mean_per_class_error", 
                        "rmse", "mse", "mae", "rmsle"
                    }
                    for metric in metrics_to_log:
                        if metric in row and pd.notna(row[metric]):
                            mlflow.log_metric(metric, row[metric])
                    
                    # Log model ID as a parameter/tag
                    mlflow.log_param("model_id", row["model_id"])
                    mlflow.set_tag("model_family", row["model_id"].split('_')[0])
                    run_ids.append(run.info.run_id)

            result_data = {"experiment_id": experiment_id, "run_ids": run_ids}
            return TaskResult(status="success", data=[Media(value=result_data)])

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}", exc_info=True)
            return TaskResult(status="error", message=f"An error occurred while logging to MLflow: {str(e)}")

if __name__ == "__main__":
    # To run this agent, you need an MLflow tracking server running.
    # You can start one with: `mlflow server --host 127.0.0.1 --port 5000`
    agent = MLflowAgent(host="127.0.0.1", port=8009, tracking_uri="http://127.0.0.1:5000")
    agent.start() 