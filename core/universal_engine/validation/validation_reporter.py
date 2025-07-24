"""
Validation Reporter - 검증 리포터
"""

import json
from typing import Dict, Any, List
from datetime import datetime

class ValidationReporter:
    """
    테스트 검증 결과를 종합하여 리포트를 생성하는 시스템입니다.
    """

    def __init__(self):
        """ValidationReporter 초기화"""
        pass

    def generate_comprehensive_report(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        종합 검증 리포트를 생성합니다.

        Args:
            test_results: 모든 테스트 결과의 목록

        Returns:
            JSON 형태의 종합 리포트
        """
        total_functions = len(test_results)
        passed_functions = sum(1 for r in test_results if r.get("overall_status") == "passed")
        success_rate = (passed_functions / total_functions) * 100 if total_functions > 0 else 0

        report = {
            "report_generated_at": datetime.now().isoformat(),
            "summary": {
                "total_functions_tested": total_functions,
                "passed_functions": passed_functions,
                "failed_functions": total_functions - passed_functions,
                "success_rate_percentage": round(success_rate, 2),
            },
            "results": test_results
        }

        # 리포트 파일 저장
        report_filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def generate_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """
        테스트 결과에 기반한 권장사항을 생성합니다.

        Args:
            test_results: 모든 테스트 결과의 목록

        Returns:
            개선 권장사항 목록
        """
        recommendations = []
        for result in test_results:
            if result.get("overall_status") == "failed":
                func_name = result.get("function_name", "Unknown function")
                details = result.get("details", {})
                for test_type, test_result in details.items():
                    if test_result.get("status") == "failed":
                        if test_type == "connection":
                            recommendations.append(f"[{func_name}] Connection failed. Check agent's health endpoint and network accessibility.")
                        elif test_type == "parameter":
                            recommendations.append(f"[{func_name}] Parameter validation failed. Check function signature and parameter handling logic.")
                        elif test_type == "execution":
                            recommendations.append(f"[{func_name}] Function execution failed. Review the function's core logic and dependencies.")
                        elif test_type == "error_handling":
                            recommendations.append(f"[{func_name}] Error handling is not robust. Improve exception handling for invalid inputs.")
                        elif test_type == "performance":
                            recommendations.append(f"[{func_name}] Performance test failed. Optimize the function to reduce response time.")
        
        if not recommendations:
            recommendations.append("All tests passed. No immediate recommendations.")

        return recommendations

    def generate_next_steps(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        검증 결과에 기반하여 다음 단계 계획을 생성합니다.

        Args:
            test_results: 모든 테스트 결과의 목록

        Returns:
            우선순위가 지정된 다음 단계 계획
        """
        action_plan = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
        }

        for result in test_results:
            if result.get("overall_status") == "failed":
                func_name = result.get("function_name", "Unknown function")
                details = result.get("details", {})
                if details.get("connection", {}).get("status") == "failed" or \
                   details.get("execution", {}).get("status") == "failed":
                    action_plan["high_priority"].append(f"Fix critical failure in {func_name} (connection or execution).")
                elif details.get("error_handling", {}).get("status") == "failed":
                    action_plan["medium_priority"].append(f"Improve error handling for {func_name}.")
                elif details.get("performance", {}).get("status") == "failed":
                    action_plan["low_priority"].append(f"Optimize performance of {func_name}.")
        
        if not any(action_plan.values()):
            return {"summary": "All tests passed. No action required."}

        return action_plan
