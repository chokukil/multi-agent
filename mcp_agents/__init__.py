# This file makes the 'mcp_agents' directory a Python package. 

import importlib
import pkgutil

# Dynamically import all agent modules in this package to ensure they
# are registered in the agent_registry.
# This runs automatically when the 'mcp_agents' package is imported.
for _, name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f".{name}", __name__) 