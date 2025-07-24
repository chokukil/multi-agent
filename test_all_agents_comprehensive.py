#!/usr/bin/env python3
"""
Comprehensive A2A Agent Testing Script
Tests all 11 agents with A2A official implementation
"""
import asyncio
import logging
from uuid import uuid4
import httpx
from datetime import datetime
import json

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agent configurations
AGENTS = [
    {"name": "Data Cleaning", "port": 8306, "test_message": "샘플 데이터로 데이터 클리닝을 테스트해주세요"},
    {"name": "Data Loader", "port": 8307, "test_message": "CSV 파일을 로드하는 방법을 알려주세요"},
    {"name": "Data Visualization", "port": 8308, "test_message": "간단한 차트를 생성해주세요"},
    {"name": "Data Wrangling", "port": 8309, "test_message": "데이터를 필터링하는 예제를 보여주세요"},
    {"name": "Feature Engineering", "port": 8310, "test_message": "피처 엔지니어링 기본 예제를 보여주세요"},
    {"name": "SQL Data Analyst", "port": 8311, "test_message": "SQL 쿼리 작성 예제를 보여주세요"},
    {"name": "EDA Tools", "port": 8312, "test_message": "기본적인 EDA를 수행해주세요"},
    {"name": "H2O ML", "port": 8313, "test_message": "H2O AutoML 사용법을 알려주세요"},
    {"name": "MLflow Tools", "port": 8314, "test_message": "MLflow 실험 추적 방법을 알려주세요"},
    {"name": "Pandas Analyst", "port": 8210, "test_message": "Pandas 데이터프레임 조작 예제를 보여주세요"},
    {"name": "Report Generator", "port": 8316, "test_message": "간단한 리포트를 생성해주세요"}
]

async def test_agent(agent_info):
    """Test a single agent"""
    name = agent_info["name"]
    port = agent_info["port"]
    test_message = agent_info["test_message"]
    base_url = f'http://localhost:{port}'
    
    result = {
        "agent": name,
        "port": port,
        "status": "failed",
        "card_retrieved": False,
        "response_received": False,
        "error": None,
        "response_preview": None
    }
    
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        try:
            # Step 1: Get Agent Card
            logger.info(f'🔍 Testing {name} Agent at {base_url}')
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            try:
                public_card = await resolver.get_agent_card()
                result["card_retrieved"] = True
                logger.info(f'✅ {name} Agent card retrieved successfully')
            except Exception as e:
                result["error"] = f"Card retrieval failed: {str(e)}"
                logger.error(f'❌ {name} Agent card retrieval failed: {e}')
                return result
            
            # Step 2: Initialize Client
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=public_card
            )
            
            # Step 3: Send Test Message
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
            
            logger.info(f'📤 Sending test message to {name} Agent...')
            response = await client.send_message(request)
            
            # Extract response text
            response_dict = response.model_dump(mode='json', exclude_none=True)
            if 'result' in response_dict and 'parts' in response_dict['result']:
                for part in response_dict['result']['parts']:
                    if part.get('kind') == 'text':
                        response_text = part.get('text', '')
                        result["response_received"] = True
                        result["response_preview"] = response_text[:200] + "..." if len(response_text) > 200 else response_text
                        result["status"] = "success"
                        logger.info(f'✅ {name} Agent responded successfully')
                        break
            
            if not result["response_received"]:
                result["error"] = "No text response in message"
                logger.warning(f'⚠️ {name} Agent returned no text response')
                
        except httpx.ConnectError:
            result["error"] = "Connection refused - agent not running"
            logger.error(f'❌ {name} Agent connection refused (not running)')
        except Exception as e:
            result["error"] = str(e)
            logger.error(f'❌ {name} Agent test failed: {e}')
    
    return result

async def test_all_agents():
    """Test all agents concurrently"""
    logger.info("🚀 Starting comprehensive A2A agent testing...")
    logger.info(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests concurrently
    tasks = [test_agent(agent) for agent in AGENTS]
    results = await asyncio.gather(*tasks)
    
    # Generate summary
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    card_retrieved = sum(1 for r in results if r["card_retrieved"])
    response_received = sum(1 for r in results if r["response_received"])
    
    summary = {
        "test_timestamp": datetime.now().isoformat(),
        "total_agents": total,
        "successful_agents": successful,
        "success_rate": f"{(successful/total)*100:.1f}%",
        "card_retrieval_rate": f"{(card_retrieved/total)*100:.1f}%",
        "response_rate": f"{(response_received/total)*100:.1f}%",
        "detailed_results": results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("📊 A2A AGENT TEST SUMMARY")
    print("="*60)
    print(f"Total Agents Tested: {total}")
    print(f"Successful Tests: {successful} ({summary['success_rate']})")
    print(f"Card Retrieval Success: {card_retrieved} ({summary['card_retrieval_rate']})")
    print(f"Response Success: {response_received} ({summary['response_rate']})")
    print("\n📋 Agent Status:")
    print("-"*60)
    
    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"{status_emoji} {result['agent']:20} (Port {result['port']}): {result['status']}")
        if result["error"]:
            print(f"   └─ Error: {result['error']}")
    
    # Save detailed results
    output_file = f"a2a_agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return summary

if __name__ == '__main__':
    asyncio.run(test_all_agents())