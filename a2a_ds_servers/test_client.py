import asyncio
import logging
from uuid import uuid4
import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)


async def test_data_loader_agent():
    """Test the Data Loader Agent."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    base_url = 'http://localhost:8001'
    
    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        try:
            # Fetch Agent Card
            logger.info(f'Fetching agent card from: {base_url}')
            agent_card = await resolver.get_agent_card()
            logger.info('Successfully fetched agent card')
            logger.info(f'Agent: {agent_card.name}')
            logger.info(f'Description: {agent_card.description}')
            
            # Initialize A2A Client
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )
            
            # Test 1: Simple message
            print("\n=== Test 1: List Current Directory ===")
            test_message = "Please list the contents of the current directory"
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': test_message}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # A2A 응답 구조 파싱
            if hasattr(response, 'root') and response.root:
                if hasattr(response.root, 'result') and response.root.result:
                    result = response.root.result
                    # TaskStatusUpdateEvent일 수도 있음
                    if hasattr(result, 'message') and result.message:
                        if hasattr(result.message, 'parts') and result.message.parts:
                            text = result.message.parts[0].root.text
                            print("Response:", text)
                        else:
                            print("Response: TaskStatusUpdateEvent but no message parts")
                    elif hasattr(result, 'parts') and result.parts:
                        text = result.parts[0].root.text  # Part 구조: Part(root=TextPart(...))
                        print("Response:", text)
                    else:
                        print(f"Response: Result type {type(result)} - {result}")
                else:
                    print("Response: No result found")
            else:
                print("Response: Unknown structure")
            
            # Test 2: Streaming message
            print("\n=== Test 2: Streaming Message ===")
            test_message_2 = "Load data from artifacts/data/shared_dataframes/titanic.csv.pkl"
            
            send_message_payload_2 = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': test_message_2}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload_2)
            )
            
            print("Streaming response:")
            stream_response = client.send_message_streaming(streaming_request)
            
            async for chunk in stream_response:
                if hasattr(chunk, 'root') and chunk.root:
                    if hasattr(chunk.root, 'result') and chunk.root.result:
                        result = chunk.root.result
                        # TaskStatusUpdateEvent 처리
                        if hasattr(result, 'message') and result.message:
                            if hasattr(result.message, 'parts') and result.message.parts:
                                text = result.message.parts[0].root.text
                                print(f"Chunk: {text}")
                            else:
                                print(f"Chunk: TaskStatusUpdateEvent - {result.status} (no message)")
                        elif hasattr(result, 'parts') and result.parts:
                            text = result.parts[0].root.text
                            print(f"Chunk: {text}")
                        elif hasattr(result, 'status'):
                            print(f"Chunk: Status Update - {result.status}")
                        else:
                            print(f"Chunk: Unknown result type {type(result)}")
                    else:
                        print(f"Chunk: No result in {type(chunk.root)}")
                else:
                    print(f"Chunk: Unknown structure {type(chunk)}")
            
        except Exception as e:
            logger.error(f'Error during testing: {e}', exc_info=True)


if __name__ == '__main__':
    asyncio.run(test_data_loader_agent()) 