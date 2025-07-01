#!/usr/bin/env python3
import asyncio
import httpx
from a2a.client import A2AClient
from a2a.types import Message, TextPart, SendMessageRequest, MessageSendParams

async def test_orchestrator():
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            a2a_client = A2AClient(
                httpx_client=client,
                url='http://localhost:8100'
            )
            
            msg = Message(
                messageId='test_123',
                role='user',
                parts=[TextPart(text='머신러닝이 무엇인지 간단히 설명해주세요.')]
            )
            
            params = MessageSendParams(message=msg)
            request = SendMessageRequest(
                id='req_test',
                jsonrpc='2.0',
                method='message/send',
                params=params
            )
            
            print('🧪 Testing fixed orchestrator...')
            response = await a2a_client.send_message(request)
            print(f'✅ Status: {response.root.result.status.state}')
            
            if response.root.result.status.message and response.root.result.status.message.parts:
                text = response.root.result.status.message.parts[0].root.text
                print(f'📝 Response length: {len(text)} characters')
                print(f'📄 First 500 chars: {text[:500]}...')
                print(f'📄 Last 200 chars: ...{text[-200:]}')
            else:
                print('❌ No response message found')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
