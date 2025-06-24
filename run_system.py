# run_system.py

import subprocess
import time
import sys
from collections import defaultdict

# Placeholder for server configurations, will be imported from config later
AGENT_SERVER_FILES = {
    "registry_server": "a2a_servers/registry_server.py",
    "pandas_analyst_server": "a2a_servers/pandas_server.py",
    # "sql_analyst_server": "a2a_servers/sql_server.py",
}

processes = defaultdict(lambda: None)

def start_server(name, file_path):
    """Starts a server as a subprocess."""
    if processes[name] and processes[name].poll() is None:
        print(f"Server '{name}' is already running.")
        return
    print(f"Starting server '{name}' from '{file_path}'...")
    # Using sys.executable to ensure the same python environment is used
    process = subprocess.Popen([sys.executable, file_path])
    processes[name] = process
    print(f"Server '{name}' started with PID: {process.pid}")

def stop_server(name):
    """Stops a running server subprocess."""
    process = processes.get(name)
    if process and process.poll() is None:
        print(f"Stopping server '{name}' (PID: {process.pid})...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print(f"Server '{name}' stopped.")
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"Server '{name}' forcefully killed.")
    else:
        print(f"Server '{name}' is not running or already stopped.")

def main():
    try:
        # Start all defined servers
        for name, file_path in AGENT_SERVER_FILES.items():
            start_server(name, file_path)

        # Keep the main script alive to manage subprocesses
        while True:
            time.sleep(10)
            # Here we can add logic to monitor servers and restart if they crash

    except KeyboardInterrupt:
        print("\nShutting down all servers...")
        for name in list(processes.keys()):
            stop_server(name)
        print("All servers have been shut down. Exiting.")

if __name__ == "__main__":
    main() 