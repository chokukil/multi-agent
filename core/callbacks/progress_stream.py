# core/callbacks/progress_stream.py
import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProgressStreamManager:
    """
    Manages broadcasting asynchronous updates to multiple UI clients (queues).
    This acts as a singleton to be used by backend processes to send
    progress updates to any listening frontend components.
    """
    def __init__(self):
        self._queues: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def register_queue(self, queue: asyncio.Queue):
        """Adds a new queue to the list of listeners."""
        async with self._lock:
            self._queues.append(queue)
            logger.info(f"Queue registered. Total listeners: {len(self._queues)}")

    async def unregister_queue(self, queue: asyncio.Queue):
        """Removes a queue from the list of listeners."""
        async with self._lock:
            try:
                self._queues.remove(queue)
                logger.info(f"Queue unregistered. Total listeners: {len(self._queues)}")
            except ValueError:
                logger.warning("Attempted to unregister a queue that was not registered.")

    async def stream_update(self, update: Dict[str, Any]):
        """Puts a new update into all registered queues."""
        if not self._queues:
            return

        async with self._lock:
            for queue in self._queues:
                await queue.put(update)

# A global instance of the manager for easy access across the application
progress_stream_manager = ProgressStreamManager()

__all__ = ["progress_stream_manager"] 