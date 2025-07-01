#!/usr/bin/env python3
import asyncio
import httpx
from a2a.client import A2AClient
from a2a.types import Message, TextPart, SendMessageRequest, MessageSendParams

async def test_complex_orchestrator():
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            a2a_client = A2AClient(
                httpx_client=client,
                url='http://localhost:8100'
            )
            
            complex_request = """
            당신은 20년 경력의 반도체 이온주입 공정(Process) 엔지니어입니다.
            다음 도메인 지식들을 숙지하고, 입력된 LOT 히스토리, 공정 계측값, 장비 정보 및 레시피 셋팅 데이터를 기반으로 
            공정 이상 여부를 판단하고, 그 원인을 설명하며, 기술적 조치 방향을 제안하는 역할을 수행합니다.
            
            이온주입 공정에서 TW(Taper Width) 이상 원인을 분석해주세요.
            """
            
            msg = Message(
                messageId='test_complex_123',
                role='user',
                parts=[TextPart(text=complex_request)]
            )
            
            params = MessageSendParams(message=msg)
            request = SendMessageRequest(
                id='req_complex_test',
                jsonrpc='2.0',
                method='message/send',
                params=params
            )
            
            print('🧪 Testing complex analysis request...')
            response = await a2a_client.send_message(request)
            print(f'✅ Status: {response.root.result.status.state}')
            
            if response.root.result.status.message and response.root.result.status.message.parts:
                text = response.root.result.status.message.parts[0].root.text
                print(f'📝 Response length: {len(text)} characters')
                print(f'📄 First 800 chars: {text[:800]}...')
                
                # 에이전트 실행 여부 확인
                if 'AI_DS_Team' in text:
                    print('✅ Multi-agent execution detected!')
                elif '실행' in text and '단계' in text:
                    print('✅ Complex processing detected!')
                else:
                    print('ℹ️  Simple processing used')
            else:
                print('❌ No response message found')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complex_orchestrator())
