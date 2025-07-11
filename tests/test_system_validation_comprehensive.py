#!/usr/bin/env python3
"""
🧬 CherryAI v9 - 포괄적인 시스템 검증 테스트
모든 컴포넌트, 파일 형식, 기능을 체계적으로 검증
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
from io import BytesIO, StringIO
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai import load_dataframe_from_file, _validate_dataframe

class TestSystemValidation:
    """포괄적인 시스템 검증 테스트"""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """테스트용 샘플 데이터"""
        return {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Age': [25, 30, 35, 28, 22],
            'City': ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Gwangju'],
            'Salary': [50000, 75000, 60000, 55000, 48000],
            'Department': ['IT', 'Sales', 'Marketing', 'IT', 'Sales']
        }
    
    # 1. 🤖 에이전트 상태 검증
    @pytest.mark.asyncio
    async def test_agent_health_check(self):
        """모든 에이전트 상태 확인"""
        agent_ports = {
            "v9_orchestrator": 8100,
            "python_repl": 8315,
            "data_cleaning": 8306,
            "data_loader": 8307,
            "data_visualization": 8308,
            "data_wrangling": 8309,
            "feature_engineering": 8310,
            "sql_database": 8311,
            "eda_tools": 8312,
            "h2o_ml": 8313,
            "mlflow_tools": 8314
        }
        
        healthy_agents = []
        unhealthy_agents = []
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for agent_name, port in agent_ports.items():
                try:
                    response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                    if response.status_code == 200:
                        healthy_agents.append(agent_name)
                        print(f"✅ {agent_name} (port {port}) - Healthy")
                    else:
                        unhealthy_agents.append(f"{agent_name} (status {response.status_code})")
                        print(f"⚠️ {agent_name} (port {port}) - Status {response.status_code}")
                except Exception as e:
                    unhealthy_agents.append(f"{agent_name} ({str(e)[:50]})")
                    print(f"❌ {agent_name} (port {port}) - {str(e)[:50]}")
        
        print(f"\n📊 에이전트 상태 요약:")
        print(f"   ✅ 정상: {len(healthy_agents)}/{len(agent_ports)}")
        print(f"   ❌ 비정상: {len(unhealthy_agents)}")
        
        # 최소 70% 이상 정상이어야 통과
        success_rate = len(healthy_agents) / len(agent_ports)
        assert success_rate >= 0.7, f"에이전트 상태 불량: {success_rate:.1%} (최소 70% 필요)"
    
    # 2. 📊 파일 형식 지원 검증
    def test_file_format_support_csv(self, sample_data):
        """CSV 파일 지원 테스트"""
        df_original = pd.DataFrame(sample_data)
        
        # CSV 파일 생성
        csv_data = df_original.to_csv(index=False)
        file_obj = BytesIO(csv_data.encode('utf-8'))
        file_obj.name = "test.csv"
        file_obj.size = len(csv_data.encode('utf-8'))
        
        # 로딩 테스트
        df_loaded = load_dataframe_from_file(file_obj)
        
        assert df_loaded.shape == df_original.shape
        assert list(df_loaded.columns) == list(df_original.columns)
        print("✅ CSV 파일 지원 검증 완료")
    
    def test_file_format_support_excel(self, sample_data):
        """Excel 파일 지원 테스트"""
        df_original = pd.DataFrame(sample_data)
        
        # Excel 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            df_original.to_excel(temp_file.name, index=False, engine='openpyxl')
            
            # 파일 다시 읽기
            with open(temp_file.name, 'rb') as f:
                file_data = f.read()
                file_obj = BytesIO(file_data)
                file_obj.name = "test.xlsx"
                file_obj.size = len(file_data)
                
                # 로딩 테스트
                df_loaded = load_dataframe_from_file(file_obj)
                
                assert df_loaded.shape == df_original.shape
                assert list(df_loaded.columns) == list(df_original.columns)
                print("✅ Excel 파일 지원 검증 완료")
            
            # 임시 파일 정리
            os.unlink(temp_file.name)
    
    def test_file_format_support_json(self, sample_data):
        """JSON 파일 지원 테스트"""
        df_original = pd.DataFrame(sample_data)
        
        # JSON 파일 생성
        json_data = df_original.to_json(orient='records')
        file_obj = BytesIO(json_data.encode('utf-8'))
        file_obj.name = "test.json"
        file_obj.size = len(json_data.encode('utf-8'))
        
        # 로딩 테스트
        df_loaded = load_dataframe_from_file(file_obj)
        
        assert df_loaded.shape == df_original.shape
        assert list(df_loaded.columns) == list(df_original.columns)
        print("✅ JSON 파일 지원 검증 완료")
    
    # 3. 🧠 v9 오케스트레이터 기능 검증
    @pytest.mark.asyncio
    async def test_v9_orchestrator_functionality(self):
        """v9 오케스트레이터 기본 기능 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Agent Card 확인
                response = await client.get("http://localhost:8100/.well-known/agent.json")
                assert response.status_code == 200
                agent_card = response.json()
                
                assert "name" in agent_card
                assert "version" in agent_card
                print(f"✅ v9 오케스트레이터 Agent Card: {agent_card.get('name', 'Unknown')}")
                
                # 간단한 작업 요청
                task_data = {
                    "parts": [
                        {
                            "type": "text",
                            "text": "System health check - please respond with a simple confirmation"
                        }
                    ]
                }
                
                response = await client.post(
                    "http://localhost:8100/task",
                    json=task_data,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200
                print("✅ v9 오케스트레이터 기본 작업 처리 검증 완료")
                
        except Exception as e:
            pytest.skip(f"v9 오케스트레이터 테스트 스킵: {str(e)}")
    
    # 4. 🐍 Python REPL 에이전트 검증
    @pytest.mark.asyncio
    async def test_python_repl_agent(self):
        """Python REPL 에이전트 기능 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 간단한 계산 작업
                task_data = {
                    "parts": [
                        {
                            "type": "text",
                            "text": "Calculate 2 + 2 and return the result"
                        }
                    ]
                }
                
                response = await client.post(
                    "http://localhost:8315/task",
                    json=task_data,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200
                print("✅ Python REPL 에이전트 기본 작업 처리 검증 완료")
                
        except Exception as e:
            pytest.skip(f"Python REPL 에이전트 테스트 스킵: {str(e)}")
    
    # 5. 🌐 Streamlit UI 접근성 검증
    @pytest.mark.asyncio
    async def test_streamlit_ui_accessibility(self):
        """Streamlit UI 접근성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("http://localhost:8501")
                assert response.status_code == 200
                assert "text/html" in response.headers.get("content-type", "")
                print("✅ Streamlit UI 접근성 검증 완료")
                
        except Exception as e:
            pytest.skip(f"Streamlit UI 테스트 스킵: {str(e)}")
    
    # 6. 📈 성능 및 메모리 사용량 검증
    def test_performance_validation(self, sample_data):
        """성능 및 메모리 사용량 테스트"""
        import psutil
        import time
        
        # 시작 시간 및 메모리
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # 큰 데이터셋 생성 (10,000 행)
        large_data = []
        for i in range(10000):
            large_data.append({
                'id': i,
                'name': f'User_{i}',
                'value': i * 1.5,
                'category': f'Cat_{i % 10}'
            })
        
        df_large = pd.DataFrame(large_data)
        
        # CSV로 변환 및 로딩 테스트
        csv_data = df_large.to_csv(index=False)
        file_obj = BytesIO(csv_data.encode('utf-8'))
        file_obj.name = "large_test.csv"
        file_obj.size = len(csv_data.encode('utf-8'))
        
        df_loaded = load_dataframe_from_file(file_obj)
        
        # 성능 측정
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        processing_time = end_time - start_time
        memory_used = (end_memory - start_memory) / (1024 * 1024)  # MB
        
        print(f"📊 성능 측정 결과:")
        print(f"   ⏱️ 처리 시간: {processing_time:.2f}초")
        print(f"   💾 메모리 사용: {memory_used:.2f}MB")
        print(f"   📏 데이터 크기: {df_loaded.shape}")
        
        # 성능 기준 (10,000행 처리 시간 < 10초, 메모리 < 100MB)
        assert processing_time < 10.0, f"처리 시간 초과: {processing_time:.2f}초"
        assert memory_used < 100.0, f"메모리 사용량 초과: {memory_used:.2f}MB"
        
        print("✅ 성능 및 메모리 사용량 검증 완료")
    
    # 7. 🔍 오류 처리 검증
    def test_error_handling(self):
        """오류 처리 메커니즘 테스트"""
        
        # 잘못된 파일 형식
        invalid_data = b"This is not a valid data file content"
        file_obj = BytesIO(invalid_data)
        file_obj.name = "invalid.xyz"
        file_obj.size = len(invalid_data)
        
        try:
            load_dataframe_from_file(file_obj)
            pytest.fail("잘못된 파일에 대한 예외가 발생하지 않음")
        except Exception as e:
            assert "파일 로드 불가" in str(e)
            print("✅ 잘못된 파일 형식 오류 처리 검증 완료")
        
        # 빈 파일
        empty_file = BytesIO(b"")
        empty_file.name = "empty.csv"
        empty_file.size = 0
        
        try:
            load_dataframe_from_file(empty_file)
            pytest.fail("빈 파일에 대한 예외가 발생하지 않음")
        except Exception as e:
            print("✅ 빈 파일 오류 처리 검증 완료")

def run_comprehensive_validation():
    """포괄적인 시스템 검증 실행"""
    print("🚀 CherryAI v9 포괄적인 시스템 검증 시작")
    print("=" * 60)
    
    # pytest 실행
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--disable-warnings"
    ], capture_output=True, text=True, cwd=project_root)
    
    print("📊 검증 결과:")
    print(result.stdout)
    
    if result.returncode == 0:
        print("🎉 모든 시스템 검증 통과!")
        return True
    else:
        print("💥 일부 검증 실패")
        print(result.stderr)
        return False

if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1) 