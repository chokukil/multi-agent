"""
Comprehensive Function Tester - 포괄적인 기능 테스터
"""

import asyncio
import aiohttp
import time
from typing import Dict, Any, List

class ComprehensiveFunctionTester:
    """
    에이전트의 기능을 포괄적으로 테스트하는 시스템입니다.
    5단계 테스트 프로세스를 통해 기능의 안정성과 품질을 검증합니다.
    """

    def __init__(self):
        """ComprehensiveFunctionTester 초기화"""
        pass

    async def test_function(self, agent_info: Dict[str, Any], function_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        주어진 함수에 대해 5단계 테스트를 수행합니다.

        Args:
            agent_info: 테스트할 에이전트의 정보
            function_info: 테스트할 함수의 정보

        Returns:
            테스트 결과 요약
        """
        test_results = {}

        # 1. 기본 연결 테스트
        test_results['connection'] = await self.test_basic_connection(agent_info)

        # 2. 파라미터 검증 테스트
        test_results['parameter'] = await self.test_parameters(function_info)

        # 3. 실제 기능 실행 테스트
        test_results['execution'] = await self.test_function_execution(agent_info, function_info)

        # 4. 에러 처리 테스트
        test_results['error_handling'] = await self.test_error_handling(agent_info, function_info)

        # 5. 성능 테스트
        test_results['performance'] = await self.test_performance(agent_info, function_info)

        return {
            "function_name": function_info.get("name"),
            "overall_status": "passed" if all(r.get("status") == "passed" for r in test_results.values()) else "failed",
            "details": test_results
        }

    async def test_basic_connection(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """기본 연결 테스트"""
        health_url = f"{agent_info.get('base_url')}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        return {"status": "passed", "details": f"Successfully connected to {health_url}"}
                    else:
                        return {"status": "failed", "details": f"Connection to {health_url} failed with status {response.status}"}
        except Exception as e:
            return {"status": "failed", "details": f"Connection to {health_url} failed with exception: {e}"}

    async def test_parameters(self, function_info: Dict[str, Any]) -> Dict[str, Any]:
        """파라미터 검증 테스트"""
        parameters = function_info.get("parameters")
        if parameters is None:
            return {"status": "failed", "details": "Parameter information not found."}

        if not isinstance(parameters, (list, str)):
            return {"status": "failed", "details": f"Invalid parameter format: {type(parameters)}"}

        # discover_from_api_endpoints 에서 온 경우
        if isinstance(parameters, list):
            required_params = [p for p in parameters if p.get("required")]
            if not required_params:
                return {"status": "passed", "details": "No required parameters found, or all are optional."}
            # 실제 호출 테스트는 execution 단계에서 수행
            return {"status": "passed", "details": f"Found {len(required_params)} required parameters."}

        # discover_from_implementation 에서 온 경우 (문자열)
        elif isinstance(parameters, str):
            if not parameters:
                return {"status": "passed", "details": "Function has no parameters."}
            # 'self' 외에 다른 파라미터가 있는지 확인
            params_list = [p.strip() for p in parameters.split(',')]
            if len(params_list) <= 1 and 'self' in params_list:
                 return {"status": "passed", "details": "Function only has 'self' parameter."}
            return {"status": "passed", "details": f"Function has parameters: {parameters}"}

        return {"status": "failed", "details": "Could not validate parameters."}

    async def test_function_execution(self, agent_info: Dict[str, Any], function_info: Dict[str, Any]) -> Dict[str, Any]:
        """실제 기능 실행 테스트"""
        try:
            test_data = await self._prepare_test_data(function_info)
            result = await self._call_agent_function(agent_info, function_info, test_data)

            if result.get("status") == "success":
                return {"status": "passed", "details": "Function executed successfully.", "result": result}
            else:
                return {"status": "failed", "details": "Function execution returned an error.", "result": result}

        except Exception as e:
            return {"status": "failed", "details": f"Function execution failed with exception: {e}"}

    async def _prepare_test_data(self, function_info: Dict[str, Any]) -> Dict[str, Any]:
        """테스트 데이터 준비"""
        test_data = {}
        parameters = function_info.get("parameters")

        if isinstance(parameters, list):  # API 엔드포인트에서 온 경우
            for param in parameters:
                param_name = param.get("name")
                param_schema = param.get("schema", {})
                param_type = param_schema.get("type")

                if param_name:
                    if param_type == "string":
                        test_data[param_name] = "test_string"
                    elif param_type == "integer":
                        test_data[param_name] = 123
                    elif param_type == "boolean":
                        test_data[param_name] = True
                    elif param_type == "array":
                        test_data[param_name] = []
                    elif param_type == "object":
                        test_data[param_name] = {}
                    else:
                        test_data[param_name] = "dummy_value"
        elif isinstance(parameters, str):  # 구현 파일에서 온 경우
            # 파라미터 문자열을 파싱하여 더미 데이터 생성
            param_names = [p.strip().split(':')[0].strip() for p in parameters.split(',') if p.strip() and p.strip() != 'self']
            for param_name in param_names:
                test_data[param_name] = "dummy_value"

        return test_data

    async def _call_agent_function(self, agent_info: Dict[str, Any], function_info: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 클라이언트를 통해 에이전트 기능 호출"""
        url = f"{agent_info.get('base_url')}{function_info.get('path')}"
        method = function_info.get("method", "POST").lower()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, json=data, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "error", "details": f"Request failed with status {response.status}", "response_text": await response.text()}
        except Exception as e:
            return {"status": "error", "details": f"Request failed with exception: {e}"}

    async def test_error_handling(self, agent_info: Dict[str, Any], function_info: Dict[str, Any]) -> Dict[str, Any]:
        """에러 처리 테스트"""
        test_cases = {
            "missing_required_parameter": {"status": "pending"},
            "invalid_parameter_type": {"status": "pending"},
        }

        # 1. 필수 파라미터 누락 테스트
        try:
            # TODO: 필수 파라미터를 의도적으로 누락시킨 요청 데이터 생성
            invalid_data = {}
            result = await self._call_agent_function(agent_info, function_info, invalid_data)
            if result.get("status") == "error":
                test_cases["missing_required_parameter"] = {"status": "passed", "details": "Correctly handled missing parameter."}
            else:
                test_cases["missing_required_parameter"] = {"status": "failed", "details": "Did not return an error for missing parameter."}
        except Exception as e:
            test_cases["missing_required_parameter"] = {"status": "passed", "details": f"Correctly raised exception for missing parameter: {e}"}

        # 2. 잘못된 파라미터 타입 테스트
        try:
            # TODO: 파라미터에 잘못된 타입의 데이터 생성
            invalid_data = {}
            result = await self._call_agent_function(agent_info, function_info, invalid_data)
            if result.get("status") == "error":
                test_cases["invalid_parameter_type"] = {"status": "passed", "details": "Correctly handled invalid parameter type."}
            else:
                test_cases["invalid_parameter_type"] = {"status": "failed", "details": "Did not return an error for invalid parameter type."}
        except Exception as e:
            test_cases["invalid_parameter_type"] = {"status": "passed", "details": f"Correctly raised exception for invalid type: {e}"}

        overall_status = "passed" if all(tc["status"] == "passed" for tc in test_cases.values()) else "failed"
        return {"status": overall_status, "details": test_cases}

    async def test_performance(self, agent_info: Dict[str, Any], function_info: Dict[str, Any]) -> Dict[str, Any]:
        """성능 테스트"""
        try:
            start_time = time.time()
            # 여러 번 실행하여 평균 응답 시간 측정
            num_runs = 3
            response_times = []
            for _ in range(num_runs):
                run_start_time = time.time()
                await self._call_agent_function(agent_info, function_info, {})
                run_end_time = time.time()
                response_times.append(run_end_time - run_start_time)
            
            avg_response_time = sum(response_times) / num_runs

            # 성능 기준 (예: 1초) 대비 평가
            if avg_response_time < 1.0:
                return {"status": "passed", "details": f"Average response time: {avg_response_time:.4f}s"}
            else:
                return {"status": "failed", "details": f"Response time ({avg_response_time:.4f}s) exceeds threshold (1.0s)"}

        except Exception as e:
            return {"status": "failed", "details": f"Performance test failed with exception: {e}"}
