#!/usr/bin/env python3
"""
Knowledge Bank Agent 완전 기능 검증 테스트
파일: a2a_ds_servers/knowledge_bank_server.py
포트: 8325
"""

import asyncio
import json
import logging
import sys
import requests
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from a2a.client import A2AClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBankValidator:
    """Knowledge Bank Agent 완전 기능 검증기"""
    
    def __init__(self):
        self.client = A2AClient()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "agent": "Knowledge Bank Server",
            "file": "a2a_ds_servers/knowledge_bank_server.py",
            "port": 8325,
            "tests": {}
        }
    
    async def test_basic_connection(self):
        """기본 연결 테스트"""
        logger.info("📋 테스트: 기본 연결")
        try:
            # Agent Card 확인
            response = requests.get("http://localhost:8325/.well-known/agent.json", timeout=5)
            if response.status_code == 200:
                card = response.json()
                logger.info("✅ Agent Card 확인 성공")
                return True
            else:
                logger.error("❌ Agent Card 확인 실패")
                return False
        except Exception as e:
            logger.error(f"❌ 기본 연결 테스트 오류: {e}")
            return False
    
    async def test_knowledge_storage(self):
        """지식 저장 기능 테스트"""
        logger.info("📋 테스트: 지식 저장 기능")
        try:
            # CSV 데이터로 지식 저장 테스트
            csv_data = """title,content,category
AI Technology,Artificial Intelligence is transforming industries,Technology
Data Science,Data analysis and machine learning techniques,Science
Business Strategy,Strategic planning and market analysis,Business"""
            
            response = await self.client.send_message(
                "http://localhost:8325",
                messageId="test_storage_001",
                role="user",
                parts=[{"kind": "text", "text": f"다음 지식을 저장해주세요:\n{csv_data}"}]
            )
            
            if response and hasattr(response, 'root') and response.root.result:
                logger.info("✅ 지식 저장 기능 성공")
                return True
            else:
                logger.error("❌ 지식 저장 기능 실패")
                return False
        except Exception as e:
            logger.error(f"❌ 지식 저장 기능 테스트 오류: {e}")
            return False
    
    async def test_knowledge_search(self):
        """지식 검색 기능 테스트"""
        logger.info("📋 테스트: 지식 검색 기능")
        try:
            # 검색 쿼리 테스트
            response = await self.client.send_message(
                "http://localhost:8325",
                messageId="test_search_001",
                role="user",
                parts=[{"kind": "text", "text": "AI Technology에 대한 지식을 검색해주세요"}]
            )
            
            if response and hasattr(response, 'root') and response.root.result:
                logger.info("✅ 지식 검색 기능 성공")
                return True
            else:
                logger.error("❌ 지식 검색 기능 실패")
                return False
        except Exception as e:
            logger.error(f"❌ 지식 검색 기능 테스트 오류: {e}")
            return False
    
    async def test_json_data_processing(self):
        """JSON 데이터 처리 테스트"""
        logger.info("📋 테스트: JSON 데이터 처리")
        try:
            # JSON 데이터로 테스트
            json_data = {
                "knowledge_items": [
                    {"title": "Machine Learning", "content": "ML algorithms and applications", "category": "AI"},
                    {"title": "Deep Learning", "content": "Neural networks and deep architectures", "category": "AI"}
                ]
            }
            
            response = await self.client.send_message(
                "http://localhost:8325",
                messageId="test_json_001",
                role="user",
                parts=[{"kind": "text", "text": f"다음 JSON 데이터를 처리해주세요: {json.dumps(json_data)}"}]
            )
            
            if response and hasattr(response, 'root') and response.root.result:
                logger.info("✅ JSON 데이터 처리 성공")
                return True
            else:
                logger.error("❌ JSON 데이터 처리 실패")
                return False
        except Exception as e:
            logger.error(f"❌ JSON 데이터 처리 테스트 오류: {e}")
            return False
    
    async def test_sample_data_generation(self):
        """샘플 데이터 생성 테스트"""
        logger.info("📋 테스트: 샘플 데이터 생성")
        try:
            # 샘플 데이터 요청
            response = await self.client.send_message(
                "http://localhost:8325",
                messageId="test_sample_001",
                role="user",
                parts=[{"kind": "text", "text": "샘플 지식 데이터를 생성해주세요"}]
            )
            
            if response and hasattr(response, 'root') and response.root.result:
                logger.info("✅ 샘플 데이터 생성 성공")
                return True
            else:
                logger.error("❌ 샘플 데이터 생성 실패")
                return False
        except Exception as e:
            logger.error(f"❌ 샘플 데이터 생성 테스트 오류: {e}")
            return False
    
    async def test_error_handling(self):
        """오류 처리 테스트"""
        logger.info("📋 테스트: 오류 처리")
        try:
            # 잘못된 데이터로 테스트
            response = await self.client.send_message(
                "http://localhost:8325",
                messageId="test_error_001",
                role="user",
                parts=[{"kind": "text", "text": "invalid data format test"}]
            )
            
            # 오류가 적절히 처리되면 성공
            if response:
                logger.info("✅ 오류 처리 성공")
                return True
            else:
                logger.error("❌ 오류 처리 실패")
                return False
        except Exception as e:
            logger.error(f"❌ 오류 처리 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🔍 Knowledge Bank Server 완전 검증 시작...")
        
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("지식 저장 기능", self.test_knowledge_storage),
            ("지식 검색 기능", self.test_knowledge_search),
            ("JSON 데이터 처리", self.test_json_data_processing),
            ("샘플 데이터 생성", self.test_sample_data_generation),
            ("오류 처리", self.test_error_handling)
        ]
        
        success_count = 0
        total_count = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 테스트: {test_name}")
            try:
                result = await test_func()
                self.results["tests"][test_name] = result
                if result:
                    logger.info(f"  결과: ✅ 성공")
                    success_count += 1
                else:
                    logger.info(f"  결과: ❌ 실패")
            except Exception as e:
                logger.error(f"  결과: ❌ 오류 - {e}")
                self.results["tests"][test_name] = False
        
        # 결과 요약
        logger.info(f"\n📊 **검증 결과 요약**:")
        logger.info(f"  성공: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        for test_name, _ in tests:
            if test_name in self.results["tests"]:
                status = "✅ 성공" if self.results["tests"][test_name] else "❌ 실패"
                logger.info(f"  {status} {test_name}")
        
        # 결과 저장
        filename = f"knowledge_bank_validation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")
        
        return success_count == total_count

async def main():
    """메인 함수"""
    validator = KnowledgeBankValidator()
    success = await validator.run_all_tests()
    
    if success:
        logger.info("🎉 모든 테스트 통과!")
    else:
        logger.error("❌ 일부 테스트 실패")

if __name__ == "__main__":
    asyncio.run(main()) 