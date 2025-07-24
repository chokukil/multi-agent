"""
Existing Agent Function Validator - 기존 에이전트 기능 검증기
"""

import asyncio
from typing import Dict, Any, List

from core.universal_engine.a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
from core.universal_engine.validation.comprehensive_function_tester import ComprehensiveFunctionTester
from core.universal_engine.validation.validation_reporter import ValidationReporter

class ExistingAgentFunctionValidator:
    """
    기존에 구현된 A2A 에이전트의 모든 기능을 검증하는 시스템입니다.
    """

    def __init__(self):
        """ExistingAgentFunctionValidator 초기화"""
        self.agent_discovery = A2AAgentDiscoverySystem()
        self.function_tester = ComprehensiveFunctionTester()
        self.reporter = ValidationReporter()

    async def discover_and_validate_all_agents(self) -> Dict[str, Any]:
        """
        모든 에이전트를 발견하고, 각 에이전트의 모든 기능을 검증합니다.

        Returns:
            종합 검증 리포트
        """
        await self.agent_discovery.start_discovery()
        all_agents = self.agent_discovery.get_available_agents()
        all_test_results = []

        for agent_id, agent_info in all_agents.items():
            # API 엔드포인트로부터 기능 발견
            functions = await self.agent_discovery.discover_from_api_endpoints(agent_info.base_url)
            
            # 구현 파일로부터 기능 발견 (보조)
            # agent_file_path = f"a2a_ds_servers/{agent_info.name.lower().replace(' ', '_')}_server.py"
            # functions.extend(await self.agent_discovery.discover_from_implementation(agent_file_path))

            for function_info in functions:
                test_result = await self.function_tester.test_function(agent_info.__dict__, function_info)
                all_test_results.append(test_result)

        await self.agent_discovery.stop_discovery()

        # 리포트 생성
        comprehensive_report = self.reporter.generate_comprehensive_report(all_test_results)
        recommendations = self.reporter.generate_recommendations(all_test_results)
        next_steps = self.reporter.generate_next_steps(all_test_results)

        comprehensive_report["recommendations"] = recommendations
        comprehensive_report["next_steps"] = next_steps

        return comprehensive_report
