import asyncio
import socket
from contextlib import closing

import pytest_asyncio
from uvicorn import Config, Server

from a2a_servers.registry_server import app


REGISTRY_HOST = "127.0.0.1"


class TestServer(Server):
    def __init__(self, app, host, port):
        self._startup_done = asyncio.Event()
        super().__init__(config=Config(app, host=host, port=port))

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        self._startup_done.set()

    async def _serve(self):
        try:
            await self.serve()
        except asyncio.CancelledError:
            await self.shutdown()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest_asyncio.fixture(scope="session")
async def registry_server():
    port = find_free_port()
    server = TestServer(app=app, host=REGISTRY_HOST, port=port)
    server_task = asyncio.create_task(server._serve())
    await server._startup_done.wait()

    yield f"http://{REGISTRY_HOST}:{port}"

    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
