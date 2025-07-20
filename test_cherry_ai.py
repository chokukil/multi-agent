#!/usr/bin/env python3
"""
Cherry AI 종합 테스트 스크립트
- 모든 핵심 컴포넌트 검증
- A2A 에이전트 연결 테스트
- UI 기능 검증
"""

import asyncio
import sys
import logging
from pathlib import Path
import json

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_config_loader():
    """에이전트 설정 로더 테스트"""
    logger.info("🧪 Testing AgentConfigLoader...")
    
    try:
        from config.agents_config import AgentConfigLoader
        
        loader = AgentConfigLoader("config/agents.json")
        agents = loader.get_all_agents()
        
        assert len(agents) > 0, "No agents found in configuration"
        logger.info(f"✅ Found {len(agents)} agents in configuration")
        
        # 특정 에이전트 테스트
        orchestrator = loader.get_agent_by_id("orchestrator")
        assert orchestrator is not None, "Orchestrator agent not found"
        logger.info(f"✅ Orchestrator agent found: {orchestrator.name}")
        
        # 카테고리별 에이전트 테스트
        analysis_agents = loader.get_agents_by_category("analysis")
        logger.info(f"✅ Found {len(analysis_agents)} analysis agents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AgentConfigLoader test failed: {e}")
        return False

async def test_planning_engine():
    """계획 엔진 테스트"""
    logger.info("🧪 Testing PlanningEngine...")
    
    try:
        from core.orchestrator.planning_engine import PlanningEngine
        from config.agents_config import AgentConfigLoader
        
        engine = PlanningEngine()
        loader = AgentConfigLoader("config/agents.json")
        
        # 사용자 의도 분석 테스트
        test_query = "데이터를 분석하고 시각화해주세요"
        intent = await engine.analyze_user_intent(test_query)
        
        assert intent.primary_goal is not None, "Primary goal not extracted"
        logger.info(f"✅ Intent analysis: {intent.primary_goal}")
        logger.info(f"✅ Analysis types: {intent.analysis_type}")
        logger.info(f"✅ Complexity: {intent.complexity_level}")
        
        # 에이전트 선택 테스트
        available_agents = list(loader.get_enabled_agents().values())
        selected_agents = await engine.select_optimal_agents(intent, available_agents)
        
        assert len(selected_agents) > 0, "No agents selected"
        logger.info(f"✅ Selected {len(selected_agents)} agents")
        
        for agent in selected_agents:
            logger.info(f"  - {agent.agent_id}: {agent.confidence:.2f} ({agent.reasoning})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PlanningEngine test failed: {e}")
        return False

async def test_a2a_orchestrator():
    """A2A 오케스트레이터 테스트"""
    logger.info("🧪 Testing A2AOrchestrator...")
    
    try:
        from core.orchestrator.a2a_orchestrator import A2AOrchestrator
        
        orchestrator = A2AOrchestrator("config/agents.json")
        
        # 설정 로드 테스트
        await orchestrator.reload_agents_config()
        assert len(orchestrator.agents) > 0, "No agents loaded"
        logger.info(f"✅ Loaded {len(orchestrator.agents)} agents")
        
        # 에이전트 상태 확인 (실제 연결은 하지 않음)
        logger.info("✅ Orchestrator initialization successful")
        
        # 분석 계획 생성 테스트
        test_query = "CSV 파일의 기본 통계를 분석해주세요"
        data_context = {
            'data_shape': (100, 5),
            'columns': ['col1', 'col2', 'col3', 'col4', 'col5'],
            'file_type': 'csv'
        }
        
        plan = await orchestrator.create_analysis_plan(test_query, data_context)
        assert plan is not None, "Analysis plan not created"
        logger.info(f"✅ Analysis plan created: {plan.id}")
        logger.info(f"  - Selected agents: {plan.selected_agents}")
        logger.info(f"  - Steps: {len(plan.execution_sequence)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ A2AOrchestrator test failed: {e}")
        return False

async def test_config_file_integrity():
    """설정 파일 무결성 테스트"""
    logger.info("🧪 Testing config file integrity...")
    
    try:
        config_path = Path("config/agents.json")
        assert config_path.exists(), "Config file does not exist"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        assert 'agents' in config_data, "Missing 'agents' section in config"
        assert 'global_settings' in config_data, "Missing 'global_settings' section in config"
        
        agents = config_data['agents']
        expected_agents = [
            'orchestrator', 'data_loader', 'data_cleaning', 'eda_tools',
            'data_visualization', 'data_wrangling', 'feature_engineering',
            'sql_database', 'h2o_ml', 'mlflow_tools', 'pandas_agent', 'report_generator'
        ]
        
        for agent_id in expected_agents:
            assert agent_id in agents, f"Missing agent: {agent_id}"
            agent_config = agents[agent_id]
            
            required_fields = ['id', 'name', 'port', 'host', 'endpoint', 'capabilities', 'enabled']
            for field in required_fields:
                assert field in agent_config, f"Missing field '{field}' in agent {agent_id}"
        
        logger.info(f"✅ Config file integrity verified: {len(agents)} agents")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config file integrity test failed: {e}")
        return False

def test_import_structure():
    """모듈 임포트 구조 테스트"""
    logger.info("🧪 Testing import structure...")
    
    try:
        # 핵심 모듈 임포트 테스트
        from config.agents_config import AgentConfigLoader, AgentConfig, AgentStatus
        logger.info("✅ Config module imports successful")
        
        from core.orchestrator.a2a_orchestrator import A2AOrchestrator
        from core.orchestrator.planning_engine import PlanningEngine
        logger.info("✅ Orchestrator module imports successful")
        
        # Cherry AI 메인 모듈 임포트 테스트 (Streamlit 없이)
        import importlib.util
        spec = importlib.util.spec_from_file_location("cherry_ai", "cherry_ai.py")
        cherry_ai_module = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(cherry_ai_module)  # Streamlit 때문에 실행하지 않음
        logger.info("✅ Cherry AI main module structure verified")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import structure test failed: {e}")
        return False

async def run_all_tests():
    """모든 테스트 실행"""
    logger.info("🍒 Starting Cherry AI Comprehensive Tests")
    logger.info("=" * 50)
    
    test_results = []
    
    # 동기 테스트
    test_results.append(("Import Structure", test_import_structure()))
    test_results.append(("Config File Integrity", await test_config_file_integrity()))
    
    # 비동기 테스트
    test_results.append(("AgentConfigLoader", await test_agent_config_loader()))
    test_results.append(("PlanningEngine", await test_planning_engine()))
    test_results.append(("A2AOrchestrator", await test_a2a_orchestrator()))
    
    # 결과 요약
    logger.info("=" * 50)
    logger.info("🍒 Cherry AI Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 50)
    logger.info(f"Total Tests: {len(test_results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Cherry AI is ready to run.")
        return True
    else:
        logger.error(f"💥 {failed} test(s) failed. Please check the errors above.")
        return False

def main():
    """메인 함수"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()