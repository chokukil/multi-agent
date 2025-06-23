import json
from python_a2a import A2AServer
from abc import ABC, abstractmethod
import logging
from fastapi import APIRouter, FastAPI

class BaseA2AAgent(ABC):
    """
    A base class for A2A agents that initializes an A2A server from a config file.
    """
    def __init__(self, config_path: str, api_router: APIRouter = None):
        """
        Initializes the agent by loading configuration from a JSON file.

        Args:
            config_path (str): The path to the agent's JSON configuration file.
            api_router (APIRouter, optional): FastAPI router for custom endpoints.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {config_path}")
            raise

        self.agent_name = config.get("agent_name", "UnnamedAgent")
        self.version = config.get("version", "1.0")
        self.description = config.get("description", "")
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 8000)

        # Combine the A2A server app with the custom API router
        self.app = FastAPI(title=self.agent_name, version=self.version, description=self.description)
        self.a2a_server = A2AServer(app=self.app)

        if api_router:
            self.app.include_router(api_router)

        self.register_skills()
        logging.info(f"Initialized agent '{self.agent_name}' on http://{self.host}:{self.port}")

    @abstractmethod
    def register_skills(self):
        """
        Abstract method for subclasses to register their skills using the
        `@self.a2a_server.skill()` decorator.
        """
        pass

    def start(self):
        """Starts the Uvicorn server to run the agent."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port) 