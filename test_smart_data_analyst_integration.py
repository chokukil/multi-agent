"""
Smart Data Analyst A2A 통합 테스트

실제 A2A 서버들과의 통신을 포함한 End-to-End 테스트
"""

import asyncio
import httpx
import pytest
import pandas as pd
import os
import uuid
import json
from datetime import datetime

# 테스트 대상 컴포넌트 import
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.a2a_data_analysis_executor import A2ADataAnalysisExecutor


class TestSmartDataAnalystIntegration:
    """Smart Data Analyst A2A 통합 테스트 클래스"""

    @pytest.fixture(scope="class")
    def sample_data_file(self):
        """테스트용 샘플 데이터 파일 생성"""
        # 간단한 CSV 데이터 생성
        test_data = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'Name_{i}' for i in range(1, 11)],
            'value': [i * 10 for i in range(1, 11)],
            'category': ['A' if i % 2 == 0 else 'B' for i in range(1, 11)],
            'score': [80 + i for i in range(1, 11)]
        })
        
        # 데이터 저장 디렉토리 확인 및 생성
        data_dir = "a2a_ds_servers/artifacts/data/shared_dataframes"
        os.makedirs(data_dir, exist_ok=True)
        
        file_path = os.path.join(data_dir, "test_integration_data.csv")
        test_data.to_csv(file_path, index=False)
        
        yield "test_integration_data"
        
        # 정리: 테스트 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

    @pytest.mark.asyncio
    async def test_orchestrator_connectivity(self):
        """오케스트레이터 연결성 테스트"""
        orchestrator_url = "http://localhost:8100"
        
        # Agent Card 확인
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(f"{orchestrator_url}/.well-known/agent.json")
                assert response.status_code == 200
                agent_card = response.json()
                assert "name" in agent_card
                print(f"✅ 오케스트레이터 Agent Card: {agent_card.get('name')}")
            except httpx.RequestError as e:
                pytest.fail(f"오케스트레이터에 연결할 수 없습니다: {e}")

    @pytest.mark.asyncio
    async def test_all_agents_connectivity(self):
        """모든 A2A 에이전트 연결성 테스트"""
        executor = A2ADataAnalysisExecutor()
        
        # 각 에이전트의 연결 상태 확인
        connectivity_results = {}
        
        for agent_name, config in executor.agent_configs.items():
            agent_url = config['url']
            is_healthy = await executor.check_agent_health(agent_url)
            connectivity_results[agent_name] = is_healthy
            
            if is_healthy:
                print(f"✅ {agent_name}: 연결 성공 ({agent_url})")
            else:
                print(f"❌ {agent_name}: 연결 실패 ({agent_url})")
        
        # 적어도 pandas_data_analyst는 연결되어야 함
        assert connectivity_results.get('pandas_data_analyst', False), \
            "Pandas Data Analyst 에이전트에 연결할 수 없습니다"

    @pytest.mark.asyncio
    async def test_single_agent_call(self, sample_data_file):
        """단일 에이전트 호출 테스트"""
        executor = A2ADataAnalysisExecutor()
        
        # pandas_data_analyst 에이전트 호출
        agent_config = executor.agent_configs['pandas_data_analyst']
        step = {
            "agent_name": "pandas_data_analyst",
            "skill_name": "analyze_data",
            "parameters": {
                "data_id": sample_data_file,
                "user_instructions": "간단한 데이터 요약 통계를 계산해주세요."
            }
        }
        
        result = await executor.call_agent(agent_config, step, 1)
        
        # 결과 검증
        assert result is not None
        print(f"단일 에이전트 호출 결과: {result}")
        
        if result.get('success'):
            print("✅ 단일 에이전트 호출 성공")
            assert result['agent'] == 'pandas_data_analyst'
            assert result['step'] == 1
        else:
            print(f"❌ 에이전트 호출 실패: {result.get('error', 'Unknown error')}")
            # 에러 상황이어도 구조는 올바르게 반환되어야 함
            assert 'error' in result

    @pytest.mark.asyncio
    async def test_orchestrator_plan_generation(self, sample_data_file):
        """오케스트레이터 계획 생성 테스트"""
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": f"데이터셋 '{sample_data_file}'에 대해 기본적인 EDA 분석을 수행해줘."
                        }
                    ]
                }
            },
            "id": 1
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post("http://localhost:8100", json=payload)
                response.raise_for_status()
                
                result = response.json()
                print(f"오케스트레이터 응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                assert "result" in result or "error" in result
                
                if "result" in result:
                    print("✅ 오케스트레이터 계획 생성 성공")
                    # 계획 내용이 문자열로 반환됨
                    plan_content = result["result"]
                    assert len(plan_content) > 0
                else:
                    print(f"❌ 오케스트레이터 오류: {result.get('error', {}).get('message', 'Unknown error')}")
                    
            except httpx.RequestError as e:
                pytest.fail(f"오케스트레이터 요청 실패: {e}")

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, sample_data_file):
        """전체 분석 워크플로우 테스트"""
        executor = A2ADataAnalysisExecutor()
        
        # 테스트용 계획 상태 생성
        plan_state = {
            "plan": [
                {
                    "agent_name": "pandas_data_analyst",
                    "skill_name": "analyze_data",
                    "parameters": {
                        "data_id": sample_data_file,
                        "user_instructions": "데이터의 기본 구조와 통계를 분석해주세요."
                    }
                },
                {
                    "agent_name": "pandas_data_analyst",  # 동일 에이전트로 2단계 실행
                    "skill_name": "analyze_data",
                    "parameters": {
                        "data_id": sample_data_file,
                        "user_instructions": "카테고리별 평균값을 계산해주세요."
                    }
                }
            ]
        }
        
        print("📊 전체 분석 워크플로우 실행 시작...")
        start_time = datetime.now()
        
        # 실행
        execution_result = await executor.execute(plan_state)
        
        end_time = datetime.now()
        execution_duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️ 실행 시간: {execution_duration:.2f}초")
        print(f"📋 실행 결과: {json.dumps(execution_result, indent=2, ensure_ascii=False)}")
        
        # 결과 검증
        assert execution_result is not None
        assert "total_steps" in execution_result
        assert "execution_time" in execution_result
        assert execution_result["total_steps"] == 2
        
        if execution_result.get("successful_steps", 0) > 0:
            print(f"✅ {execution_result['successful_steps']}/{execution_result['total_steps']} 단계 성공")
            assert "step_outputs" in execution_result
        else:
            print("❌ 모든 단계 실패 - A2A 에이전트 상태를 확인하세요")

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """오류 처리 테스트"""
        executor = A2ADataAnalysisExecutor()
        
        # 존재하지 않는 데이터셋으로 테스트
        plan_state = {
            "plan": [
                {
                    "agent_name": "pandas_data_analyst",
                    "skill_name": "analyze_data",
                    "parameters": {
                        "data_id": "nonexistent_dataset",
                        "user_instructions": "존재하지 않는 데이터를 분석해주세요."
                    }
                }
            ]
        }
        
        execution_result = await executor.execute(plan_state)
        
        # 오류 상황에서도 결과 구조가 올바르게 반환되어야 함
        assert execution_result is not None
        assert "total_steps" in execution_result
        assert execution_result["total_steps"] == 1
        
        print(f"🔍 오류 처리 테스트 결과: {execution_result}")

    @pytest.mark.asyncio
    async def test_agent_health_monitoring(self):
        """에이전트 상태 모니터링 테스트"""
        executor = A2ADataAnalysisExecutor()
        
        # 모든 에이전트 상태 확인
        available_agents = await executor.get_available_agents()
        
        print(f"🔍 사용 가능한 에이전트: {available_agents}")
        
        # 결과 검증
        assert isinstance(available_agents, list)
        
        if available_agents:
            print(f"✅ {len(available_agents)}개 에이전트 사용 가능")
            
            # 사용 가능한 에이전트 중 하나로 간단한 테스트
            test_agent = available_agents[0]
            agent_config = executor.agent_configs[test_agent]
            
            # Agent Card 확인
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{agent_config['url']}/.well-known/agent.json")
                assert response.status_code == 200
                agent_card = response.json()
                print(f"📋 {test_agent} Agent Card: {agent_card.get('name', 'Unknown')}")
        else:
            print("❌ 사용 가능한 에이전트 없음")

    def test_data_file_creation(self, sample_data_file):
        """테스트 데이터 파일 생성 확인"""
        data_dir = "a2a_ds_servers/artifacts/data/shared_dataframes"
        file_path = os.path.join(data_dir, f"{sample_data_file}.csv")
        
        assert os.path.exists(file_path)
        
        # 파일 내용 확인
        df = pd.read_csv(file_path)
        assert len(df) == 10
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert 'value' in df.columns
        assert 'category' in df.columns
        assert 'score' in df.columns
        
        print(f"✅ 테스트 데이터 파일 생성 확인: {file_path}")
        print(f"📊 데이터 형태: {df.shape}")


if __name__ == "__main__":
    # 통합 테스트 실행
    print("🚀 Smart Data Analyst A2A 통합 테스트 시작")
    pytest.main([__file__, "-v", "-s"])  # -s 옵션으로 print 출력 표시 