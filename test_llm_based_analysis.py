#!/usr/bin/env python3
"""
LLM 기반 분석 로직 테스트
Rule 기반 조건문 대신 LLM이 지시사항을 이해하고 적절한 분석을 선택하는지 확인
"""

import asyncio
import httpx
import uuid
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendMessageRequest

async def test_llm_based_analysis():
    """LLM이 각기 다른 지시사항을 이해하고 적절한 분석을 수행하는지 테스트"""
    
    # 다양한 분석 요청 (rule 없이도 LLM이 이해해야 함)
    test_cases = [
        {
            "step": 1,
            "instruction": "Show me the basic structure and missing data information for this dataset",
            "expected_focus": "데이터 구조"
        },
        {
            "step": 2, 
            "instruction": "I need statistical summaries and distributions of numerical variables",
            "expected_focus": "통계 요약"
        },
        {
            "step": 3,
            "instruction": "Analyze how different variables relate to each other",
            "expected_focus": "변수 관계"
        },
        {
            "step": 4,
            "instruction": "What survival patterns can you find in this data?",
            "expected_focus": "생존 패턴"
        },
        {
            "step": 5,
            "instruction": "Give me the key takeaways and actionable insights",
            "expected_focus": "핵심 인사이트"
        }
    ]
    
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url="http://localhost:10001",
        )
        agent_card = await resolver.get_agent_card()
        client = A2AClient(
            httpx_client=httpx_client, 
            agent_card=agent_card
        )
        
        print("🤖 LLM 기반 분석 테스트 시작\n")
        print("📝 각 요청에 대해 LLM이 적절한 분석을 선택하는지 확인합니다...\n")
        
        for test_case in test_cases:
            print(f"🎯 Test {test_case['step']}: {test_case['expected_focus']}")
            print(f"💭 요청: {test_case['instruction']}")
            
            # 메시지 전송
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': test_case['instruction']}
                    ],
                    'messageId': uuid.uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid.uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # 응답 처리
            response_dict = response.model_dump(mode='json', exclude_none=True)
            content = ""
            
            if "result" in response_dict:
                result = response_dict["result"]
                if "parts" in result:
                    for part in result["parts"]:
                        if part.get("kind") == "text" or part.get("type") == "text":
                            content += part.get("text", "")
            
            if content:
                lines = content.split('\n')
                title = lines[0] if lines else "제목 없음"
                content_preview = ' '.join(content.split()[:20]) + "..." if len(content.split()) > 20 else content
                
                print(f"📊 LLM 응답 제목: {title}")
                print(f"📝 내용 미리보기: {content_preview}")
                print(f"📏 응답 길이: {len(content)} 문자")
                
                # LLM이 요청을 제대로 이해했는지 확인
                if test_case['step'] == 1 and any(keyword in content.lower() for keyword in ["structure", "구조", "missing", "결측", "기본 정보"]):
                    print("✅ LLM이 데이터 구조 분석 요청을 정확히 이해함")
                elif test_case['step'] == 2 and any(keyword in content.lower() for keyword in ["statistical", "통계", "distribution", "분포", "mean", "평균"]):
                    print("✅ LLM이 통계 분석 요청을 정확히 이해함")
                elif test_case['step'] == 3 and any(keyword in content.lower() for keyword in ["relationship", "관계", "correlation", "상관", "relate"]):
                    print("✅ LLM이 변수 관계 분석 요청을 정확히 이해함")
                elif test_case['step'] == 4 and any(keyword in content.lower() for keyword in ["survival", "생존", "pattern", "패턴"]):
                    print("✅ LLM이 생존 패턴 분석 요청을 정확히 이해함")
                elif test_case['step'] == 5 and any(keyword in content.lower() for keyword in ["insights", "인사이트", "takeaway", "핵심", "key"]):
                    print("✅ LLM이 인사이트 요약 요청을 정확히 이해함")
                else:
                    print("🤔 LLM의 분석 방향 확인 필요")
                    
            else:
                print("❌ LLM 응답 없음")
            
            print("=" * 80)
            print()

if __name__ == "__main__":
    asyncio.run(test_llm_based_analysis()) 