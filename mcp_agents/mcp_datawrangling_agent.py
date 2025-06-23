from core.agents.base_a2a_agent import BaseA2AAgent, logger
from core.data_manager import DataManager
from a2a.sdk.types import AgentCard, Skill, Task, TaskResult, Media
import pandas as pd
from typing import List, Dict

class DataWranglingAgent(BaseA2AAgent):
    """
    An agent expert in data wrangling and transformation tasks like
    merging, joining, and aggregating dataframes.
    """

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="DataWranglingAgent",
            description="Performs data transformations like merging and aggregation.",
            version="1.0.0",
            skills=[
                Skill(
                    name="merge_dataframes",
                    description="Merges two dataframes based on a common column.",
                    parameters={
                        "left_data_id": "string",
                        "right_data_id": "string",
                        "on": "string",
                        "how": "string (e.g., 'inner', 'outer', 'left', 'right')"
                    },
                    returns="string (new_data_id)",
                ),
                Skill(
                    name="aggregate_data",
                    description="Performs a groupby and aggregation.",
                    parameters={
                        "data_id": "string",
                        "group_by_cols": "array[string]",
                        "agg_funcs": "object (e.g., {'col1': 'sum', 'col2': 'mean'})"
                    },
                    returns="string (new_data_id)",
                )
            ],
        )

    def _register_skills(self):
        self.a2a_server.skill("merge_dataframes")(self.merge_dataframes)
        self.a2a_server.skill("aggregate_data")(self.aggregate_data)

    async def merge_dataframes(self, task: Task) -> TaskResult:
        left_id = task.parameters.get("left_data_id")
        right_id = task.parameters.get("right_data_id")
        on_col = task.parameters.get("on")
        how = task.parameters.get("how", "inner")

        if not all([left_id, right_id, on_col]):
            return TaskResult(status="error", message="Missing 'left_data_id', 'right_data_id', or 'on' parameter.")

        logger.info(f"DataWranglingAgent: Merging {left_id} and {right_id} on '{on_col}'")
        try:
            dm = DataManager()
            left_df = dm.get_dataframe(left_id)
            right_df = dm.get_dataframe(right_id)

            if left_df is None or right_df is None:
                return TaskResult(status="error", message="One or both data_ids not found.")

            merged_df = pd.merge(left_df, right_df, on=on_col, how=how)
            source = f"merged_{left_id}_{right_id}"
            new_id = dm.add_dataframe(merged_df, source=source)
            
            return TaskResult(status="success", data=[Media(value=new_id)])
        except Exception as e:
            logger.error(f"Error merging dataframes: {e}")
            return TaskResult(status="error", message=f"An error occurred during merge: {str(e)}")

    async def aggregate_data(self, task: Task) -> TaskResult:
        data_id = task.parameters.get("data_id")
        group_by_cols = task.parameters.get("group_by_cols")
        agg_funcs = task.parameters.get("agg_funcs")

        if not all([data_id, group_by_cols, agg_funcs]):
            return TaskResult(status="error", message="Missing 'data_id', 'group_by_cols', or 'agg_funcs' parameter.")

        logger.info(f"DataWranglingAgent: Aggregating {data_id} by {group_by_cols}")
        try:
            dm = DataManager()
            df = dm.get_dataframe(data_id)
            if df is None:
                return TaskResult(status="error", message=f"Data ID '{data_id}' not found.")
            
            aggregated_df = df.groupby(group_by_cols).agg(agg_funcs).reset_index()
            source = f"aggregated_{data_id}"
            new_id = dm.add_dataframe(aggregated_df, source=source)

            return TaskResult(status="success", data=[Media(value=new_id)])
        except Exception as e:
            logger.error(f"Error aggregating dataframe: {e}")
            return TaskResult(status="error", message=f"An error occurred during aggregation: {str(e)}")


if __name__ == "__main__":
    agent = DataWranglingAgent(host="127.0.0.1", port=8004)
    agent.start() 