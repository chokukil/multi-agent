"""
Run Validation Script
"""

import asyncio
import json

from core.universal_engine.validation.existing_agent_function_validator import ExistingAgentFunctionValidator

async def main():
    """메인 검증 실행 함수"""
    validator = ExistingAgentFunctionValidator()
    report = await validator.discover_and_validate_all_agents()

    print("--- Validation Report Summary ---")
    print(json.dumps(report["summary"], indent=2))
    print("---------------------------------")

if __name__ == "__main__":
    asyncio.run(main())
