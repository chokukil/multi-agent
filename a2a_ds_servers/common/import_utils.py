"""
Import 유틸리티 모듈
모든 A2A 서버에서 사용하는 표준 import 패턴 및 유틸리티
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Any, Tuple

logger = logging.getLogger(__name__)

def setup_project_paths() -> None:
    """
    프로젝트 경로 설정 - 모든 서버에서 동일하게 사용
    ai_data_science_team 패키지가 루트에 위치하므로 단순한 설정만 필요
    """
    project_root = Path(__file__).parent.parent.parent
    project_root_str = str(project_root)
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        logger.info(f"✅ 프로젝트 루트 경로 추가: {project_root_str}")

def safe_import_ai_ds_team(module_path: str) -> Tuple[bool, Optional[Any]]:
    """
    AI DS Team 모듈 안전 import
    
    Args:
        module_path: ai_data_science_team 하위 모듈 경로 (예: "tools.dataframe")
    
    Returns:
        (성공 여부, 모듈 객체 또는 None)
    """
    try:
        full_module_path = f"ai_data_science_team.{module_path}"
        module = __import__(full_module_path, fromlist=[''])
        logger.info(f"✅ AI DS Team 모듈 import 성공: {module_path}")
        return True, module
    except ImportError as e:
        logger.warning(f"⚠️ AI DS Team 모듈 import 실패: {module_path} - {e}")
        return False, None
    except Exception as e:
        logger.error(f"❌ AI DS Team 모듈 import 오류: {module_path} - {e}")
        return False, None

def get_ai_ds_agent(agent_name: str) -> Tuple[bool, Optional[Any]]:
    """
    AI DS Team 에이전트 가져오기
    
    Args:
        agent_name: 에이전트 클래스명 (예: "DataCleaningAgent")
    
    Returns:
        (성공 여부, 에이전트 클래스 또는 None)
    """
    success, agents_module = safe_import_ai_ds_team("agents")
    if success and hasattr(agents_module, agent_name):
        agent_class = getattr(agents_module, agent_name)
        logger.info(f"✅ AI DS Team 에이전트 로드 성공: {agent_name}")
        return True, agent_class
    
    logger.warning(f"⚠️ AI DS Team 에이전트 로드 실패: {agent_name}")
    return False, None

def get_ai_ds_function(module_path: str, function_name: str) -> Tuple[bool, Optional[Any]]:
    """
    AI DS Team 함수 가져오기
    
    Args:
        module_path: 모듈 경로 (예: "tools.dataframe")
        function_name: 함수명 (예: "get_dataframe_summary")
    
    Returns:
        (성공 여부, 함수 객체 또는 None)
    """
    success, module = safe_import_ai_ds_team(module_path)
    if success and hasattr(module, function_name):
        function = getattr(module, function_name)
        logger.info(f"✅ AI DS Team 함수 로드 성공: {module_path}.{function_name}")
        return True, function
    
    logger.warning(f"⚠️ AI DS Team 함수 로드 실패: {module_path}.{function_name}")
    return False, None

def check_ai_ds_team_availability() -> dict:
    """
    AI DS Team 패키지 사용 가능성 체크
    
    Returns:
        체크 결과 딕셔너리
    """
    results = {
        "available": False,
        "modules": {},
        "agents": {},
        "tools": {}
    }
    
    # 기본 패키지 체크
    success, _ = safe_import_ai_ds_team("")
    results["available"] = success
    
    if success:
        # 주요 모듈들 체크
        modules_to_check = ["agents", "tools", "templates", "utils"]
        for module in modules_to_check:
            module_success, _ = safe_import_ai_ds_team(module)
            results["modules"][module] = module_success
        
        # 주요 에이전트들 체크
        agents_to_check = ["DataCleaningAgent", "DataVisualizationAgent", "EDAToolsAgent"]
        for agent in agents_to_check:
            agent_success, _ = get_ai_ds_agent(agent)
            results["agents"][agent] = agent_success
        
        # 주요 도구들 체크
        tools_to_check = [
            ("tools.dataframe", "get_dataframe_summary"),
            ("tools.eda", "explain_data"),
        ]
        for module_path, func_name in tools_to_check:
            tool_success, _ = get_ai_ds_function(module_path, func_name)
            results["tools"][f"{module_path}.{func_name}"] = tool_success
    
    return results

def log_import_status():
    """AI DS Team import 상태 로깅"""
    logger.info("🔍 AI DS Team 패키지 상태 체크 시작")
    status = check_ai_ds_team_availability()
    
    if status["available"]:
        logger.info("✅ AI DS Team 패키지 사용 가능")
        
        # 모듈 상태
        for module, available in status["modules"].items():
            status_icon = "✅" if available else "❌"
            logger.info(f"  {status_icon} 모듈 {module}: {'사용 가능' if available else '사용 불가'}")
        
        # 에이전트 상태  
        for agent, available in status["agents"].items():
            status_icon = "✅" if available else "❌"
            logger.info(f"  {status_icon} 에이전트 {agent}: {'사용 가능' if available else '사용 불가'}")
            
        # 도구 상태
        for tool, available in status["tools"].items():
            status_icon = "✅" if available else "❌"
            logger.info(f"  {status_icon} 도구 {tool}: {'사용 가능' if available else '사용 불가'}")
    else:
        logger.error("❌ AI DS Team 패키지 사용 불가")
    
    return status