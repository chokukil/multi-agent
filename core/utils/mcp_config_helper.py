# File: core/utils/mcp_config_helper.py
# Location: ./core/utils/mcp_config_helper.py

"""
MCP 설정 관리 헬퍼 모듈
multi_agent_supervisor.py의 패턴을 참고하여 MCP 설정을 생성하고 관리
Plan-Execute 패턴에 최적화된 역할별 전문 도구 할당
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

def get_default_mcp_servers() -> Dict[str, Dict[str, Any]]:
    """기본 MCP 서버 설정 반환 - mcp_config.py와 동일한 포트 사용"""
    return {
        # 기본 유틸리티 서버들 (아직 구현되지 않은 서버들)
        "task_manager": {
            "url": "http://localhost:8001/sse", 
            "transport": "sse",
            "description": "Task management and coordination"
        },
        "self_critic": {
            "url": "http://localhost:8002/sse", 
            "transport": "sse",
            "description": "Self-criticism and quality control"
        },
        "memory_kv": {
            "url": "http://localhost:8003/sse", 
            "transport": "sse",
            "description": "Key-value memory storage"
        },
        "result_ranker": {
            "url": "http://localhost:8004/sse", 
            "transport": "sse",
            "description": "Result ranking and evaluation"
        },
        "logger": {
            "url": "http://localhost:8005/sse", 
            "transport": "sse",
            "description": "Advanced logging and reporting"
        },
        
        # 실제 구현된 MCP 서버들 (mcp_config.py 포트와 일치)
        "file_management": {
            "url": "http://localhost:8006/sse", 
            "transport": "sse",
            "description": "Safe file operations and management"
        },
        "data_science_tools": {
            "url": "http://localhost:8007/sse", 
            "transport": "sse",
            "description": "Comprehensive data science tools"
        },
        "semiconductor_yield_analysis": {
            "url": "http://localhost:8008/sse", 
            "transport": "sse",
            "description": "Semiconductor yield analysis"
        },
        "process_control_charts": {
            "url": "http://localhost:8009/sse", 
            "transport": "sse",
            "description": "Process control charts"
        },
        "semiconductor_equipment_analysis": {
            "url": "http://localhost:8010/sse", 
            "transport": "sse",
            "description": "Equipment analysis"
        },
        "defect_pattern_analysis": {
            "url": "http://localhost:8011/sse", 
            "transport": "sse",
            "description": "Defect pattern analysis"
        },
        "process_optimization": {
            "url": "http://localhost:8012/sse", 
            "transport": "sse",
            "description": "Process optimization"
        },
        "timeseries_analysis": {
            "url": "http://localhost:8013/sse", 
            "transport": "sse",
            "description": "Time series analysis and forecasting"
        },
        "anomaly_detection": {
            "url": "http://localhost:8014/sse", 
            "transport": "sse",
            "description": "Anomaly detection and outlier analysis"
        },
        "advanced_ml_tools": {
            "url": "http://localhost:8016/sse", 
            "transport": "sse",
            "description": "Advanced machine learning algorithms"
        },
        "data_preprocessing_tools": {
            "url": "http://localhost:8017/sse", 
            "transport": "sse",
            "description": "Advanced data preprocessing and cleaning"
        },
        "statistical_analysis_tools": {
            "url": "http://localhost:8018/sse", 
            "transport": "sse",
            "description": "Statistical analysis and hypothesis testing"
        },
        "report_writing_tools": {
            "url": "http://localhost:8019/sse", 
            "transport": "sse",
            "description": "Professional report generation"
        },
        "semiconductor_process_tools": {
            "url": "http://localhost:8020/sse", 
            "transport": "sse",
            "description": "Comprehensive semiconductor process analysis tools"
        }
    }

def get_optimized_role_mapping() -> Dict[str, List[str]]:
    """Plan-Execute 패턴에 최적화된 역할별 MCP 도구 매핑"""
    return {
        # 데이터 검증 및 품질 관리 전문가
        "Data_Validator": [
            "data_preprocessing_tools",      # 데이터 품질 검사
            "statistical_analysis_tools",    # 기본 통계 검증
            "anomaly_detection"             # 이상값 검출
        ],
        
        # 데이터 전처리 및 특성 엔지니어링 전문가  
        "Preprocessing_Expert": [
            "data_preprocessing_tools",      # 전처리 전문 도구
            "advanced_ml_tools",            # 특성 엔지니어링
            "anomaly_detection",            # 이상치 처리
            "file_management"               # 파일 관리
        ],
        
        # 탐색적 데이터 분석 전문가
        "EDA_Analyst": [
            "data_science_tools",           # 기본 분석 도구
            "statistical_analysis_tools",   # 통계 분석
            "anomaly_detection",            # 패턴 발견
            "data_preprocessing_tools"      # 데이터 탐색
        ],
        
        # 데이터 시각화 전문가
        "Visualization_Expert": [
            "data_science_tools",           # 시각화 도구
            "statistical_analysis_tools",   # 통계 그래프
            "timeseries_analysis"           # 시계열 시각화
        ],
        
        # 머신러닝 전문가
        "ML_Specialist": [
            "advanced_ml_tools",            # ML 알고리즘
            "data_science_tools",           # 기본 도구
            "statistical_analysis_tools",   # 모델 평가
            "data_preprocessing_tools"      # 특성 처리
        ],
        
        # 통계 분석 전문가
        "Statistical_Analyst": [
            "statistical_analysis_tools",   # 전문 통계 도구
            "timeseries_analysis",          # 시계열 분석
            "data_science_tools",           # 기본 도구
            "anomaly_detection"             # 통계적 이상 검출
        ],
        
        # 보고서 생성 전문가
        "Report_Generator": [
            "report_writing_tools",         # 보고서 생성
            "file_management",              # 파일 관리
            "data_science_tools"            # 결과 시각화
        ]
    }

def get_role_descriptions() -> Dict[str, str]:
    """역할별 전문성 설명"""
    return {
        "Data_Validator": "데이터 품질 검증 및 무결성 확인 전문가",
        "Preprocessing_Expert": "데이터 전처리 및 특성 엔지니어링 전문가", 
        "EDA_Analyst": "탐색적 데이터 분석 및 패턴 발견 전문가",
        "Visualization_Expert": "데이터 시각화 및 인사이트 전달 전문가",
        "ML_Specialist": "머신러닝 모델링 및 예측 분석 전문가",
        "Statistical_Analyst": "통계 분석 및 가설 검정 전문가", 
        "Report_Generator": "분석 결과 문서화 및 보고서 작성 전문가"
    }

# 기존 함수들과 호환성을 위해 get_role_to_mcp_mapping은 유지하되 새로운 매핑을 반환
def get_role_to_mcp_mapping() -> Dict[str, List[str]]:
    """역할별 MCP 도구 매핑 반환 (최적화된 버전)"""
    return get_optimized_role_mapping()

def create_mcp_config_for_role(
    role_name: str, 
    available_servers: Dict[str, bool]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    역할별 MCP 설정을 multi_agent_supervisor.py 방식으로 생성
    
    Args:
        role_name: 역할 이름 (예: "Data_Validator")
        available_servers: 서버 가용성 상태
        
    Returns:
        Tuple[tools_list, mcp_config_dict]
        - tools_list: 도구 이름 리스트 (python_repl_ast 포함)
        - mcp_config_dict: initialize_mcp_tools 호환 MCP 설정
    """
    base_tools = ["python_repl_ast"]  # 모든 역할에 Python 도구 포함
    
    # 역할별 MCP 도구 매핑 가져오기
    role_mapping = get_optimized_role_mapping()
    default_servers = get_default_mcp_servers()
    
    # MCP 서버 설정 구성
    mcp_servers = {}
    
    if role_name in role_mapping:
        required_servers = role_mapping[role_name]
        
        for server_name in required_servers:
            # 서버가 사용 가능한지 확인
            if available_servers.get(server_name, False):
                if server_name in default_servers:
                    mcp_servers[server_name] = default_servers[server_name].copy()
                    logging.info(f"✅ Added MCP server '{server_name}' to {role_name}")
                else:
                    logging.warning(f"⚠️ Unknown MCP server '{server_name}' for {role_name}")
            else:
                logging.info(f"💤 MCP server '{server_name}' not available for {role_name}")
    else:
        logging.warning(f"⚠️ No MCP mapping found for role '{role_name}'")
    
    # MCP 설정 구성 (initialize_mcp_tools 호환 형태)
    mcp_config = {}
    if mcp_servers:
        mcp_config = {
            "mcpServers": mcp_servers,
            "config_name": f"{role_name.lower()}_tools",
            "description": f"MCP tools for {role_name}",
            "role": role_name
        }
        
        # 도구 리스트에 MCP 설정 이름 추가
        base_tools.append(f"mcp:{mcp_config['config_name']}")
        
        logging.info(f"🔧 Created MCP config for {role_name} with {len(mcp_servers)} servers")
    else:
        logging.info(f"🐍 {role_name} will use Python tools only (no MCP servers available)")
    
    return base_tools, mcp_config

def create_supervisor_tools_config(available_servers: Dict[str, bool]) -> Dict[str, Any]:
    """
    Supervisor용 통합 MCP 도구 설정 생성
    multi_agent_supervisor.py의 supervisor_tools 패턴을 따름
    """
    default_servers = get_default_mcp_servers()
    
    # 사용 가능한 모든 서버를 포함
    available_mcp_servers = {
        name: config for name, config in default_servers.items()
        if available_servers.get(name, False)
    }
    
    if not available_mcp_servers:
        return {}
    
    return {
        "mcpServers": available_mcp_servers,
        "config_name": "supervisor_tools",
        "description": "Comprehensive MCP tools for supervisor coordination",
        "role": "supervisor"
    }

def save_mcp_config_to_file(config_name: str, mcp_config: Dict[str, Any]) -> Optional[Path]:
    """
    MCP 설정을 JSON 파일로 저장 (multi_agent_supervisor.py 호환)
    
    Args:
        config_name: 설정 파일 이름
        mcp_config: MCP 설정 딕셔너리
        
    Returns:
        저장된 파일 경로 또는 None
    """
    try:
        # mcp-config 디렉토리 생성
        config_dir = Path("mcp-config")
        config_dir.mkdir(exist_ok=True)
        
        # 파일 경로 생성
        config_file = config_dir / f"{config_name}.json"
        
        # JSON 파일로 저장
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(mcp_config, f, ensure_ascii=False, indent=2)
        
        logging.info(f"MCP config saved to {config_file}")
        return config_file
        
    except Exception as e:
        logging.error(f"Failed to save MCP config '{config_name}': {e}")
        return None

def load_mcp_config_from_file(config_name: str) -> Optional[Dict[str, Any]]:
    """
    JSON 파일에서 MCP 설정 로드
    
    Args:
        config_name: 설정 파일 이름 (확장자 제외)
        
    Returns:
        MCP 설정 딕셔너리 또는 None
    """
    try:
        config_file = Path("mcp-config") / f"{config_name}.json"
        
        if not config_file.exists():
            logging.warning(f"MCP config file not found: {config_file}")
            return None
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        logging.info(f"MCP config loaded from {config_file}")
        return config
        
    except Exception as e:
        logging.error(f"Failed to load MCP config '{config_name}': {e}")
        return None

def validate_mcp_config(mcp_config: Dict[str, Any]) -> bool:
    """
    MCP 설정의 유효성 검증
    
    Args:
        mcp_config: 검증할 MCP 설정
        
    Returns:
        유효성 여부
    """
    if not isinstance(mcp_config, dict):
        return False
    
    # 필수 키 확인
    required_keys = ["mcpServers"]
    for key in required_keys:
        if key not in mcp_config:
            logging.error(f"Missing required key in MCP config: {key}")
            return False
    
    # mcpServers 구조 확인
    mcp_servers = mcp_config["mcpServers"]
    if not isinstance(mcp_servers, dict):
        logging.error("mcpServers must be a dictionary")
        return False
    
    # 각 서버 설정 확인
    for server_name, server_config in mcp_servers.items():
        if not isinstance(server_config, dict):
            logging.error(f"Server config for '{server_name}' must be a dictionary")
            return False
        
        # 필수 서버 설정 키 확인
        required_server_keys = ["url", "transport"]
        for key in required_server_keys:
            if key not in server_config:
                logging.error(f"Missing required key '{key}' in server config for '{server_name}'")
                return False
    
    return True

def debug_mcp_config(role_name: str, tools: List[str], mcp_config: Dict[str, Any]) -> None:
    """
    MCP 설정 디버깅 정보 출력
    
    Args:
        role_name: 역할 이름
        tools: 도구 리스트
        mcp_config: MCP 설정
    """
    logging.info(f"=== MCP Config Debug for {role_name} ===")
    logging.info(f"Tools: {tools}")
    logging.info(f"MCP Config keys: {list(mcp_config.keys())}")
    
    if "mcpServers" in mcp_config:
        servers = mcp_config["mcpServers"]
        logging.info(f"MCP Servers: {list(servers.keys())}")
        
        for server_name, server_config in servers.items():
            logging.info(f"  {server_name}: {server_config.get('url', 'No URL')}")
    else:
        logging.info("No mcpServers found in config")
    
    logging.info("=== End MCP Config Debug ===") 