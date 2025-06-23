from core.agents.base_a2a_agent import BaseA2AAgent, logger
from core.data_manager import DataManager
from a2a.sdk.types import AgentCard, Skill, Task, TaskResult, Media
import pandas as pd
import plotly.express as px
import json

class DataVisualizationAgent(BaseA2AAgent):
    """
    An agent that creates insightful visualizations from data.
    """

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="DataVisualizationAgent",
            description="Creates various plots from data and returns them as JSON objects.",
            version="1.0.0",
            skills=[
                Skill(
                    name="create_plot",
                    description="Creates a plot of a specified type.",
                    parameters={
                        "data_id": "string",
                        "plot_type": "string (e.g., 'histogram', 'scatter', 'bar', 'line', 'box')",
                        "params": "object (plotly.express parameters, e.g., {'x': 'col1', 'y': 'col2'})"
                    },
                    returns="object (Plotly JSON)",
                ),
            ],
        )

    def _register_skills(self):
        self.a2a_server.skill("create_plot")(self.create_plot)

    async def create_plot(self, task: Task) -> TaskResult:
        data_id = task.parameters.get("data_id")
        plot_type = task.parameters.get("plot_type")
        params = task.parameters.get("params", {})

        if not all([data_id, plot_type]):
            return TaskResult(status="error", message="Missing 'data_id' or 'plot_type' parameter.")

        logger.info(f"DataVisualizationAgent: Creating '{plot_type}' plot for {data_id}")
        try:
            dm = DataManager()
            df = dm.get_dataframe(data_id)
            if df is None:
                return TaskResult(status="error", message=f"Data ID '{data_id}' not found.")

            if not hasattr(px, plot_type):
                return TaskResult(status="error", message=f"Invalid plot_type: '{plot_type}'.")

            plot_func = getattr(px, plot_type)
            fig = plot_func(df, **params)
            
            # Serialize figure to JSON
            plot_json = fig.to_json()

            return TaskResult(status="success", data=[Media(value=json.loads(plot_json))])
        except Exception as e:
            logger.error(f"Error creating plot: {e}", exc_info=True)
            return TaskResult(status="error", message=f"An error occurred while creating the plot: {str(e)}")


if __name__ == "__main__":
    agent = DataVisualizationAgent(host="127.0.0.1", port=8007)
    agent.start() 