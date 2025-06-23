from core.agents.base_a2a_agent import BaseA2AAgent, logger
from core.data_manager import DataManager
from a2a.sdk.types import AgentCard, Skill, Task, TaskResult, Media
from sqlalchemy import create_engine, text
import pandas as pd
import os

class SQLDatabaseAgent(BaseA2AAgent):
    """
    An agent expert in connecting to SQL databases, executing queries,
    and loading the results into the system.
    """

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="SQLDatabaseAgent",
            description="Connects to a SQL database, executes a query, and returns a data_id.",
            version="1.0.0",
            skills=[
                Skill(
                    name="execute_query",
                    description="Executes a SQL query on a given database and returns a data_id.",
                    parameters={
                        "db_uri": "string (e.g., 'sqlite:///your_database.db')",
                        "sql_query": "string"
                    },
                    returns="string",
                ),
            ],
        )

    def _register_skills(self):
        self.a2a_server.skill("execute_query")(self.execute_query)

    async def execute_query(self, task: Task) -> TaskResult:
        db_uri = task.parameters.get("db_uri")
        sql_query = task.parameters.get("sql_query")

        if not all([db_uri, sql_query]):
            return TaskResult(status="error", message="Missing 'db_uri' or 'sql_query' parameter.")
        
        logger.info(f"SQLDatabaseAgent: Received task to execute query on {db_uri}")

        try:
            engine = create_engine(db_uri)
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql_query), connection)

            data_manager = DataManager()
            source_name = f"sql:{engine.url.database}"
            data_id = data_manager.add_dataframe(df, source=source_name)

            if data_id:
                return TaskResult(status="success", data=[Media(value=data_id)])
            else:
                return TaskResult(status="error", message="Failed to add dataframe to DataManager.")
        except Exception as e:
            logger.error(f"Error executing query on {db_uri}: {e}")
            return TaskResult(status="error", message=f"An error occurred while executing the query: {str(e)}")

if __name__ == "__main__":
    # Example of running the agent with an in-memory SQLite DB for testing
    # To test:
    # 1. Run this agent.
    # 2. Use a client to send a task to http://127.0.0.1:8002/a2a/v1/task
    #    with body: { "skill": "execute_query", "parameters": { "db_uri": "sqlite:///:memory:", "sql_query": "CREATE TABLE users(id INT, name VARCHAR); INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'); SELECT * FROM users;" } }
    # NOTE: For persistent SQLite, use "sqlite:///your_db_file.db"
    agent = SQLDatabaseAgent(host="127.0.0.1", port=8002)
    agent.start() 