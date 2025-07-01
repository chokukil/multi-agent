#!/usr/bin/env python3
"""
데이터 파일 참조 시스템 테스트
오케스트레이터가 올바른 데이터 파일을 에이전트들에게 전달하는지 확인
"""

import asyncio
import json
import logging
from a2a.client import A2AClient
from a2a.types import SendMessageRequest, TextPart, Part

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_file_reference():
    """데이터 파일 참조 테스트"""
    
    print("🔬 데이터 파일 참조 시스템 테스트 시작")
    print("=" * 60)
    
    # A2A 클라이언트 생성
    client = A2AClient("http://localhost:8301")
    
    try:
        # 테스트 요청 - 반도체 데이터 분석
        test_message = """
이 반도체 이온 주입 데이터를 분석해서 다음을 수행해주세요:

1. 데이터 로딩 및 전처리
2. TW AVG 값의 분포 분석
3. 장비별(MAIN EQP ID) 성능 비교
4. 이상치 탐지
5. 시각화 차트 생성
6. 종합 분석 보고서

특히 어떤 데이터 파일을 사용하는지 각 단계에서 명확히 표시해주세요.
"""
        
        print(f"📤 테스트 요청 전송:")
        print(f"   메시지: {test_message[:100]}...")
        
        # A2A 요청 생성
        request = SendMessageRequest(
            message=Part(root=TextPart(text=test_message)),
            messageId=f"data_file_test_{int(asyncio.get_event_loop().time())}"
        )
        
        print("🔄 오케스트레이터 응답 대기 중...")
        
        # 스트리밍 응답 처리
        data_files_mentioned = []
        step_count = 0
        
        async for chunk in client.stream_message(request):
            if hasattr(chunk, 'content'):
                content = chunk.content
                
                # 데이터 파일 언급 확인
                if any(keyword in content.lower() for keyword in ['ion_implant', '.csv', '.xlsx', 'data_file', '데이터 파일']):
                    if content not in data_files_mentioned:
                        data_files_mentioned.append(content)
                        print(f"📁 데이터 파일 언급 발견: {content[:100]}...")
                
                # 단계 진행 확인
                if '단계' in content or 'step' in content.lower():
                    step_count += 1
                    print(f"📋 단계 {step_count} 진행 중...")
                
                # 에러 확인
                if any(error_keyword in content.lower() for error_keyword in ['error', 'exception', 'not defined', '오류', '실패']):
                    print(f"❌ 에러 감지: {content[:100]}...")
        
        print("\n" + "=" * 60)
        print("🔍 테스트 결과 분석:")
        print(f"   총 단계 수: {step_count}")
        print(f"   데이터 파일 언급 횟수: {len(data_files_mentioned)}")
        
        if data_files_mentioned:
            print("📁 언급된 데이터 파일들:")
            for i, mention in enumerate(data_files_mentioned[:5], 1):
                print(f"   {i}. {mention[:150]}...")
        else:
            print("⚠️ 데이터 파일 언급이 발견되지 않았습니다.")
        
        # 실제 데이터 파일 확인
        import os
        data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
        if os.path.exists(data_path):
            available_files = [f for f in os.listdir(data_path) if f.endswith(('.csv', '.pkl'))]
            print(f"\n📂 실제 사용 가능한 데이터 파일들: {available_files}")
            
            ion_implant_files = [f for f in available_files if 'ion_implant' in f.lower()]
            if ion_implant_files:
                print(f"🔬 ion_implant 파일들: {ion_implant_files}")
            else:
                print("⚠️ ion_implant 파일이 없습니다!")
        
        print("\n✅ 데이터 파일 참조 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)

async def test_specific_data_file_request():
    """특정 데이터 파일 요청 테스트"""
    
    print("\n🎯 특정 데이터 파일 요청 테스트")
    print("=" * 60)
    
    client = A2AClient("http://localhost:8301")
    
    try:
        # 특정 파일을 명시한 요청
        specific_request = """
ion_implant_3lot_dataset.csv 파일을 사용해서 다음 분석을 수행해주세요:

1. 데이터 기본 정보 확인
2. TW AVG 컬럼의 통계 분석
3. 장비별 성능 비교

반드시 ion_implant_3lot_dataset.csv 파일만 사용하고, 다른 파일은 사용하지 마세요.
"""
        
        print(f"📤 특정 파일 요청 전송:")
        print(f"   요청: {specific_request[:100]}...")
        
        request = SendMessageRequest(
            message=Part(root=TextPart(text=specific_request)),
            messageId=f"specific_file_test_{int(asyncio.get_event_loop().time())}"
        )
        
        correct_file_usage = 0
        wrong_file_usage = 0
        
        async for chunk in client.stream_message(request):
            if hasattr(chunk, 'content'):
                content = chunk.content
                
                # 올바른 파일 사용 확인
                if 'ion_implant_3lot_dataset.csv' in content:
                    correct_file_usage += 1
                
                # 잘못된 파일 사용 확인
                if any(wrong_file in content.lower() for wrong_file in ['churn_data', 'sales_data', 'employee_data']) and '.csv' in content:
                    wrong_file_usage += 1
                    print(f"⚠️ 잘못된 파일 사용 감지: {content[:100]}...")
        
        print(f"\n📊 특정 파일 요청 결과:")
        print(f"   올바른 파일 사용: {correct_file_usage}회")
        print(f"   잘못된 파일 사용: {wrong_file_usage}회")
        
        if wrong_file_usage == 0:
            print("✅ 특정 파일 요청이 올바르게 처리되었습니다!")
        else:
            print("❌ 잘못된 파일이 사용되었습니다. 데이터 파일 참조 시스템을 개선해야 합니다.")
            
    except Exception as e:
        print(f"❌ 특정 파일 테스트 실행 중 오류: {e}")

async def main():
    """메인 테스트 실행"""
    print("🧪 CherryAI 데이터 파일 참조 시스템 종합 테스트")
    print("=" * 80)
    
    # 기본 데이터 파일 참조 테스트
    await test_data_file_reference()
    
    # 특정 데이터 파일 요청 테스트
    await test_specific_data_file_request()
    
    print("\n" + "=" * 80)
    print("🏁 모든 테스트 완료")

if __name__ == "__main__":
    asyncio.run(main()) 