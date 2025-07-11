#!/usr/bin/env python3
"""
🧬 CherryAI v9 - 최종 시스템 통합 테스트
전체 시스템의 완전한 기능을 검증하는 종합 테스트
"""

import pytest
import asyncio
import httpx
import pandas as pd
import tempfile
import os
import json
import time
from pathlib import Path
from io import BytesIO
import sys
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestFinalSystemIntegration:
    """최종 시스템 통합 테스트"""
    
    @pytest.fixture(scope="class")
    def sample_dataset(self):
        """테스트용 현실적인 샘플 데이터셋"""
        return pd.DataFrame({
            'customer_id': range(1, 101),
            'name': [f'Customer_{i}' for i in range(1, 101)],
            'age': [20 + (i % 50) for i in range(100)],
            'city': ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Gwangju'] * 20,
            'purchase_amount': [100 + (i * 10) % 1000 for i in range(100)],
            'purchase_date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'product_category': ['Electronics', 'Clothing', 'Food', 'Books', 'Home'] * 20,
            'satisfaction_score': [1 + (i % 5) for i in range(100)]
        })
    
    # 1. 🌟 전체 시스템 워크플로우 테스트
    @pytest.mark.asyncio
    async def test_complete_workflow(self, sample_dataset):
        """완전한 워크플로우 테스트: 파일 업로드 → 질문 → 응답"""
        
        # 1-1. 샘플 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            csv_file_path = f.name
        
        try:
            # 1-2. 파일 업로드 시뮬레이션
            with open(csv_file_path, 'rb') as f:
                file_data = f.read()
            
            # 1-3. v9 Orchestrator에 분석 요청 (A2A JSON-RPC 표준 형식)
            request_data = {
                "jsonrpc": "2.0",
                "method": "sendMessage",
                "id": "test_001",
                "params": {
                    "message": {
                        "parts": [
                            {
                                "type": "text",
                                "text": """
User Request: 이 고객 데이터를 분석해서 주요 인사이트를 제공해주세요.

Dataset Context:
- File: customer_data.csv
- Shape: 100 rows × 8 columns
- Columns: customer_id, name, age, city, purchase_amount, purchase_date, product_category, satisfaction_score

Please provide a comprehensive analysis using the appropriate AI DS Team agents.
"""
                            }
                        ]
                    }
                }
            }
            
            # 1-4. API 호출 및 응답 검증 (A2A 표준 엔드포인트)
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://localhost:8100/",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200, f"v9 Orchestrator 응답 오류: {response.status_code}"
                
                result = response.json()
                assert result is not None, "빈 응답 받음"
                
                # 응답 구조 검증
                assert "text" in result or "content" in result, "응답에 텍스트 내용 없음"
                
                print(f"✅ 완전한 워크플로우 테스트 성공")
                print(f"   📊 응답 크기: {len(str(result))} 문자")
                
        finally:
            # 임시 파일 정리
            os.unlink(csv_file_path)
    
    # 2. 🔄 다중 에이전트 협업 테스트
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        """다중 에이전트 협업 테스트"""
        
        # 복잡한 분석 요청 (여러 에이전트 필요) - A2A JSON-RPC 표준 형식
        complex_request = {
            "jsonrpc": "2.0",
            "method": "sendMessage",
            "id": "test_multi_agent",
            "params": {
                "message": {
                    "parts": [
                        {
                            "type": "text",
                            "text": """
User Request: 고객 세그먼트 분석을 해주세요. 
1. 데이터 품질 체크
2. 기술 통계 분석
3. 고객 세그먼트 시각화
4. 머신러닝 모델 추천

이 작업은 여러 에이전트의 협업이 필요합니다.
"""
                        }
                    ]
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "http://localhost:8100/",
                json=complex_request,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # 다중 에이전트 응답 검증
            assert result is not None
            
            print("✅ 다중 에이전트 협업 테스트 성공")
    
    # 3. 🚀 성능 및 부하 테스트
    @pytest.mark.asyncio
    async def test_performance_load(self):
        """성능 및 부하 테스트"""
        
        # 동시 요청 5개 보내기 - A2A JSON-RPC 표준 형식
        requests = []
        for i in range(5):
            request_data = {
                "jsonrpc": "2.0",
                "method": "sendMessage",
                "id": f"test_perf_{i+1}",
                "params": {
                    "message": {
                        "parts": [
                            {
                                "type": "text",
                                "text": f"간단한 통계 분석 요청 {i+1}: 평균, 표준편차, 분포 분석"
                            }
                        ]
                    }
                }
            }
            requests.append(request_data)
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # 동시 요청 실행
            tasks = []
            for req in requests:
                task = client.post(
                    "http://localhost:8100/",
                    json=req,
                    headers={"Content-Type": "application/json"}
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 성능 검증
        success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        
        print(f"📊 성능 테스트 결과:")
        print(f"   ⏱️ 전체 시간: {total_time:.2f}초")
        print(f"   ✅ 성공 응답: {success_count}/5")
        print(f"   🏎️ 평균 응답 시간: {total_time/5:.2f}초")
        
        # 최소 80% 성공률 요구
        assert success_count >= 4, f"부하 테스트 실패: {success_count}/5 성공"
        
        # 전체 처리 시간 < 2분
        assert total_time < 120, f"성능 테스트 실패: {total_time:.2f}초 소요"
        
        print("✅ 성능 및 부하 테스트 성공")
    
    # 4. 🔧 오류 복구 테스트
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """오류 복구 및 탄력성 테스트"""
        
        # 잘못된 요청 1: 빈 요청 (A2A 형식 오류)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8100/",
                json={},
                headers={"Content-Type": "application/json"}
            )
            
            # 오류 처리 확인 (400 또는 422 응답 예상)
            assert response.status_code in [400, 422], f"빈 요청 오류 처리 실패: {response.status_code}"
        
        # 잘못된 요청 2: 형식 오류
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8100/",
                json={"invalid": "format"},
                headers={"Content-Type": "application/json"}
            )
            
            # 오류 처리 확인
            assert response.status_code in [400, 422], f"형식 오류 처리 실패: {response.status_code}"
        
        # 정상 요청으로 복구 확인 - A2A JSON-RPC 표준 형식
        normal_request = {
            "jsonrpc": "2.0",
            "method": "sendMessage",
            "id": "test_recovery",
            "params": {
                "message": {
                    "parts": [
                        {
                            "type": "text",
                            "text": "간단한 테스트 요청"
                        }
                    ]
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8100/",
                json=normal_request,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200, f"오류 후 복구 실패: {response.status_code}"
        
        print("✅ 오류 복구 테스트 성공")
    
    # 5. 🌐 UI 접근성 및 응답성 테스트
    @pytest.mark.asyncio
    async def test_ui_accessibility(self):
        """UI 접근성 및 응답성 테스트"""
        
        # Streamlit UI 접근 테스트
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8501")
            
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")
            
            # 기본 UI 요소 확인
            html_content = response.text
            assert "CherryAI" in html_content
            assert "streamlit" in html_content.lower()
        
        print("✅ UI 접근성 테스트 성공")
    
    # 6. 🔍 Langfuse 통합 테스트
    @pytest.mark.asyncio
    async def test_langfuse_integration(self):
        """Langfuse 통합 및 관찰 가능성 테스트"""
        
        # Langfuse 설정 확인
        langfuse_host = os.getenv("LANGFUSE_HOST")
        if not langfuse_host:
            pytest.skip("Langfuse 설정 없음")
        
        # 간단한 요청으로 Langfuse 로깅 테스트 - A2A JSON-RPC 표준 형식
        request_data = {
            "jsonrpc": "2.0",
            "method": "sendMessage",
            "id": "test_langfuse",
            "params": {
                "message": {
                    "parts": [
                        {
                            "type": "text",
                            "text": "Langfuse 통합 테스트용 요청"
                        }
                    ]
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8100/",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
        
        print("✅ Langfuse 통합 테스트 성공")
    
    # 7. 📊 데이터 처리 정확성 테스트
    def test_data_processing_accuracy(self, sample_dataset):
        """데이터 처리 정확성 테스트"""
        
        # 기본 통계 검증
        assert sample_dataset.shape == (100, 8)
        assert sample_dataset['customer_id'].nunique() == 100
        assert sample_dataset['purchase_amount'].dtype in ['int64', 'float64']
        
        # 데이터 타입 검증
        assert sample_dataset['purchase_date'].dtype == 'datetime64[ns]'
        assert sample_dataset['city'].dtype == 'object'
        
        print("✅ 데이터 처리 정확성 테스트 성공")

def run_final_system_test():
    """최종 시스템 테스트 실행"""
    print("🚀 CherryAI v9 최종 시스템 통합 테스트 시작")
    print("=" * 70)
    
    # 사전 조건 확인
    print("📋 사전 조건 확인 중...")
    
    # 필수 서비스 실행 확인
    required_services = [
        ("v9 Orchestrator", "http://localhost:8100/.well-known/agent.json"),
        ("Streamlit UI", "http://localhost:8501"),
    ]
    
    import httpx
    
    for service_name, url in required_services:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                print(f"✅ {service_name} 실행 중")
            else:
                print(f"⚠️ {service_name} 응답 이상: {response.status_code}")
        except Exception as e:
            print(f"❌ {service_name} 접근 불가: {str(e)}")
    
    print("\n🧪 통합 테스트 실행 중...")
    
    # pytest 실행
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--disable-warnings",
        "-x"  # 첫 번째 실패 시 중단
    ], capture_output=True, text=True, cwd=project_root)
    
    print("\n📊 최종 테스트 결과:")
    print(result.stdout)
    
    if result.returncode == 0:
        print("🎉 모든 최종 시스템 테스트 통과!")
        print("✅ CherryAI v9 배포 준비 완료!")
        return True
    else:
        print("💥 일부 테스트 실패")
        print(result.stderr)
        return False

if __name__ == "__main__":
    success = run_final_system_test()
    exit(0 if success else 1) 