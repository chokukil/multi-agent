import json, requests
from typing import Any, Dict, Optional

class A2AAgentClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def send_message(self, input_text: str, meta: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "messageId": f"cherry-ai-{hash(input_text) % 10000}",
                "contextId": "cherry-ai-context",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": input_text
                    }
                ]
            },
            "id": 1
        }
        r = requests.post(self.endpoint, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
