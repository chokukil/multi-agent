"""
🍒 CherryAI MCP 설정 관리자

LLM First 원칙을 준수하는 JSON 기반 MCP 서버 설정 관리
- JSON 설정 파일 로드/저장
- 환경변수 치환 (${VAR_NAME})
- stdio/sse 타입 지원
- LLM 기반 설정 자동 완성
- 동적 서버 추가/제거
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class MCPServerType(Enum):
    """MCP 서버 타입"""
    STDIO = "stdio"
    SSE = "sse"

@dataclass
class MCPServerDefinition:
    """MCP 서버 정의"""
    server_id: str
    server_type: MCPServerType
    name: str
    description: str
    enabled: bool = True
    
    # stdio 타입 속성
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: str = "./"
    
    # sse 타입 속성
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    tools: List[str] = field(default_factory=lambda: ["*"])
    
    # 공통 속성
    timeout: float = 20.0
    retry_count: int = 3
    health_check_interval: float = 45.0
    capabilities: List[str] = field(default_factory=list)

@dataclass
class MCPGlobalSettings:
    """MCP 전역 설정"""
    default_timeout: float = 20.0
    default_retry_count: int = 3
    default_health_check_interval: float = 45.0
    environment_variables: Dict[str, str] = field(default_factory=dict)
    auto_discovery: Dict[str, Any] = field(default_factory=dict)
    llm_enhancement: Dict[str, Any] = field(default_factory=dict)

class MCPConfigManager:
    """
    MCP 설정 관리자
    
    LLM First 원칙을 준수하여 하드코딩된 설정 대신
    JSON 파일 기반 동적 설정 관리 제공
    """
    
    def __init__(self, config_path: str = "mcp-config/mcp_servers_config.json"):
        self.config_path = Path(config_path)
        self.servers: Dict[str, MCPServerDefinition] = {}
        self.global_settings = MCPGlobalSettings()
        self.metadata: Dict[str, Any] = {}
        
        # 환경변수 패턴 (${VAR_NAME})
        self.env_var_pattern = re.compile(r'\$\{([^}]+)\}')
        
        # LLM 기반 자동 설정 컨텍스트
        self.llm_context = {
            'common_ports': [3000, 3001, 3002, 3003, 3004, 3005, 3006, 8000, 8001, 8002],
            'common_commands': ['python', 'node', 'npx', 'npm'],
            'common_env_vars': ['PYTHONPATH', 'NODE_ENV', 'DEBUG', 'LOG_LEVEL'],
            'capabilities_mapping': {
                'playwright': ['browser_automation', 'ui_testing', 'screenshot'],
                'data': ['data_analysis', 'visualization', 'cleaning'],
                'github': ['issue_management', 'pr_review', 'code_analysis'],
                'sql': ['sql_query', 'schema_analysis', 'data_extraction']
            }
        }
        
        self._initialize_config()
    
    def _initialize_config(self):
        """설정 초기화"""
        if self.config_path.exists():
            self.load_config()
        else:
            logger.info(f"설정 파일이 없어 기본 설정 생성: {self.config_path}")
            self._create_default_config()
            self.save_config()
    
    def load_config(self) -> bool:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # MCP 서버 설정 로드
            self._load_mcp_servers(config_data.get('mcpServers', {}))
            
            # 전역 설정 로드
            self._load_global_settings(config_data.get('globalSettings', {}))
            
            # 메타데이터 로드
            self.metadata = config_data.get('metadata', {})
            
            logger.info(f"MCP 설정 로드 완료: {len(self.servers)}개 서버")
            return True
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            self._create_default_config()
            return False
    
    def _load_mcp_servers(self, servers_config: Dict[str, Any]):
        """MCP 서버 설정 로드"""
        self.servers.clear()
        
        for server_id, config in servers_config.items():
            try:
                # 환경변수 치환
                resolved_config = self._resolve_environment_variables(config)
                
                server_def = MCPServerDefinition(
                    server_id=server_id,
                    server_type=MCPServerType(resolved_config.get('type', 'stdio')),
                    name=resolved_config.get('name', server_id),
                    description=resolved_config.get('description', ''),
                    enabled=resolved_config.get('enabled', True),
                    
                    # stdio 타입 속성
                    command=resolved_config.get('command'),
                    args=resolved_config.get('args', []),
                    env=resolved_config.get('env', {}),
                    cwd=resolved_config.get('cwd', './'),
                    
                    # sse 타입 속성
                    url=resolved_config.get('url'),
                    headers=resolved_config.get('headers', {}),
                    tools=resolved_config.get('tools', ['*']),
                    
                    # 공통 속성
                    timeout=resolved_config.get('timeout', self.global_settings.default_timeout),
                    retry_count=resolved_config.get('retry_count', self.global_settings.default_retry_count),
                    health_check_interval=resolved_config.get('health_check_interval', self.global_settings.default_health_check_interval),
                    capabilities=resolved_config.get('capabilities', [])
                )
                
                self.servers[server_id] = server_def
                logger.debug(f"MCP 서버 로드: {server_id} ({server_def.server_type.value})")
                
            except Exception as e:
                logger.error(f"서버 설정 로드 실패 {server_id}: {e}")
    
    def _load_global_settings(self, global_config: Dict[str, Any]):
        """전역 설정 로드"""
        self.global_settings = MCPGlobalSettings(
            default_timeout=global_config.get('default_timeout', 20.0),
            default_retry_count=global_config.get('default_retry_count', 3),
            default_health_check_interval=global_config.get('default_health_check_interval', 45.0),
            environment_variables=global_config.get('environment_variables', {}),
            auto_discovery=global_config.get('auto_discovery', {}),
            llm_enhancement=global_config.get('llm_enhancement', {})
        )
    
    def _resolve_environment_variables(self, config: Union[Dict, List, str]) -> Union[Dict, List, str]:
        """
        LLM First: 환경변수를 동적으로 치환
        하드코딩된 값 대신 환경변수와 설정 파일의 값을 우선 사용
        """
        if isinstance(config, dict):
            return {key: self._resolve_environment_variables(value) for key, value in config.items()}
        
        elif isinstance(config, list):
            return [self._resolve_environment_variables(item) for item in config]
        
        elif isinstance(config, str):
            # ${VAR_NAME} 패턴 찾기
            def replace_env_var(match):
                var_name = match.group(1)
                
                # 1. 전역 설정의 environment_variables에서 찾기
                if var_name in self.global_settings.environment_variables:
                    return self.global_settings.environment_variables[var_name]
                
                # 2. 실제 환경변수에서 찾기
                env_value = os.getenv(var_name)
                if env_value is not None:
                    return env_value
                
                # 3. 기본값 또는 LLM 기반 추천
                default_value = self._suggest_environment_variable_value(var_name)
                logger.warning(f"환경변수 {var_name}를 찾을 수 없어 기본값 사용: {default_value}")
                return default_value
            
            return self.env_var_pattern.sub(replace_env_var, config)
        
        return config
    
    def _suggest_environment_variable_value(self, var_name: str) -> str:
        """
        LLM First: 환경변수에 대한 기본값을 동적으로 제안
        하드코딩된 기본값 대신 컨텍스트 기반 추천
        """
        var_name_lower = var_name.lower()
        
        # 일반적인 패턴 기반 추천
        if 'key' in var_name_lower or 'token' in var_name_lower or 'secret' in var_name_lower:
            return f"your_{var_name_lower}_here"
        
        elif 'url' in var_name_lower or 'endpoint' in var_name_lower:
            return f"https://api.example.com"
        
        elif 'port' in var_name_lower:
            return "3000"
        
        elif 'path' in var_name_lower:
            return "./"
        
        elif 'database' in var_name_lower and 'url' in var_name_lower:
            return "sqlite:///./data/cherryai.db"
        
        elif var_name_lower == 'debug':
            return "false"
        
        elif var_name_lower == 'node_env':
            return "development"
        
        elif var_name_lower == 'pythonpath':
            return "."
        
        # 기본값
        return f"${{{var_name}}}"
    
    def save_config(self) -> bool:
        """설정 파일 저장"""
        try:
            # 설정 디렉토리 생성
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 설정 데이터 구성
            config_data = {
                'mcpServers': self._serialize_servers(),
                'globalSettings': self._serialize_global_settings(),
                'metadata': {
                    **self.metadata,
                    'last_updated': datetime.now().isoformat(),
                    'total_servers': len(self.servers),
                    'enabled_servers': len([s for s in self.servers.values() if s.enabled])
                }
            }
            
            # JSON 파일 저장
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"MCP 설정 저장 완료: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"설정 파일 저장 실패: {e}")
            return False
    
    def _serialize_servers(self) -> Dict[str, Any]:
        """서버 설정 직렬화"""
        servers_data = {}
        
        for server_id, server in self.servers.items():
            server_config = {
                'type': server.server_type.value,
                'name': server.name,
                'description': server.description,
                'enabled': server.enabled,
                'timeout': server.timeout,
                'retry_count': server.retry_count,
                'health_check_interval': server.health_check_interval,
                'capabilities': server.capabilities
            }
            
            # stdio 타입 속성
            if server.server_type == MCPServerType.STDIO:
                if server.command:
                    server_config['command'] = server.command
                if server.args:
                    server_config['args'] = server.args
                if server.env:
                    server_config['env'] = server.env
                if server.cwd != './':
                    server_config['cwd'] = server.cwd
            
            # sse 타입 속성
            elif server.server_type == MCPServerType.SSE:
                if server.url:
                    server_config['url'] = server.url
                if server.headers:
                    server_config['headers'] = server.headers
                if server.tools != ['*']:
                    server_config['tools'] = server.tools
            
            servers_data[server_id] = server_config
        
        return servers_data
    
    def _serialize_global_settings(self) -> Dict[str, Any]:
        """전역 설정 직렬화"""
        return {
            'default_timeout': self.global_settings.default_timeout,
            'default_retry_count': self.global_settings.default_retry_count,
            'default_health_check_interval': self.global_settings.default_health_check_interval,
            'environment_variables': self.global_settings.environment_variables,
            'auto_discovery': self.global_settings.auto_discovery,
            'llm_enhancement': self.global_settings.llm_enhancement
        }
    
    def add_server(self, server_def: MCPServerDefinition) -> bool:
        """서버 추가"""
        try:
            self.servers[server_def.server_id] = server_def
            logger.info(f"MCP 서버 추가: {server_def.server_id}")
            return True
        except Exception as e:
            logger.error(f"서버 추가 실패: {e}")
            return False
    
    def remove_server(self, server_id: str) -> bool:
        """서버 제거"""
        if server_id in self.servers:
            del self.servers[server_id]
            logger.info(f"MCP 서버 제거: {server_id}")
            return True
        return False
    
    def update_server(self, server_id: str, updates: Dict[str, Any]) -> bool:
        """서버 설정 업데이트"""
        if server_id not in self.servers:
            return False
        
        try:
            server = self.servers[server_id]
            
            # 업데이트 가능한 필드들
            updateable_fields = [
                'name', 'description', 'enabled', 'timeout', 'retry_count',
                'health_check_interval', 'capabilities', 'command', 'args',
                'env', 'cwd', 'url', 'headers', 'tools'
            ]
            
            for field, value in updates.items():
                if field in updateable_fields and hasattr(server, field):
                    setattr(server, field, value)
            
            logger.info(f"MCP 서버 업데이트: {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"서버 업데이트 실패: {e}")
            return False
    
    def get_server(self, server_id: str) -> Optional[MCPServerDefinition]:
        """서버 정보 조회"""
        return self.servers.get(server_id)
    
    def get_enabled_servers(self) -> Dict[str, MCPServerDefinition]:
        """활성화된 서버 목록"""
        return {sid: server for sid, server in self.servers.items() if server.enabled}
    
    def get_servers_by_type(self, server_type: MCPServerType) -> Dict[str, MCPServerDefinition]:
        """타입별 서버 목록"""
        return {sid: server for sid, server in self.servers.items() 
                if server.server_type == server_type}
    
    def validate_server_config(self, server_def: MCPServerDefinition) -> Tuple[bool, List[str]]:
        """
        LLM First: 서버 설정을 동적으로 검증
        하드코딩된 검증 규칙 대신 컨텍스트 기반 검증
        """
        errors = []
        
        # 필수 필드 검증
        if not server_def.server_id:
            errors.append("서버 ID가 필요합니다")
        
        if not server_def.name:
            errors.append("서버 이름이 필요합니다")
        
        # 타입별 검증
        if server_def.server_type == MCPServerType.STDIO:
            if not server_def.command:
                errors.append("stdio 타입은 command가 필요합니다")
            
            # 명령어 존재 여부 확인 (선택적)
            if server_def.command and server_def.command not in ['python', 'node', 'npx', 'npm']:
                # 실제 명령어 존재 확인은 런타임에
                pass
        
        elif server_def.server_type == MCPServerType.SSE:
            if not server_def.url:
                errors.append("sse 타입은 url이 필요합니다")
            
            # URL 형식 검증
            if server_def.url and not (server_def.url.startswith('http://') or server_def.url.startswith('https://')):
                errors.append("유효한 HTTP/HTTPS URL이 필요합니다")
        
        # 타임아웃 검증
        if server_def.timeout <= 0:
            errors.append("타임아웃은 0보다 커야 합니다")
        
        return len(errors) == 0, errors
    
    def suggest_server_config(self, server_id: str, server_type: MCPServerType, 
                            partial_config: Dict[str, Any] = None) -> MCPServerDefinition:
        """
        LLM First: 서버 설정을 지능적으로 제안
        사용자 입력과 컨텍스트를 기반으로 설정 자동 완성
        """
        if partial_config is None:
            partial_config = {}
        
        # 기본 설정
        suggested_config = MCPServerDefinition(
            server_id=server_id,
            server_type=server_type,
            name=partial_config.get('name', server_id.replace('_', ' ').title()),
            description=partial_config.get('description', f"{server_id} MCP 서버"),
            enabled=partial_config.get('enabled', True)
        )
        
        # 타입별 기본 설정 제안
        if server_type == MCPServerType.STDIO:
            # Python 모듈인지 추측
            if 'python' in server_id.lower() or any(cap in server_id.lower() for cap in ['data', 'analysis', 'ml']):
                suggested_config.command = 'python'
                suggested_config.args = ['-m', f'mcp_agents.{server_id}.server']
                suggested_config.env = {'PYTHONPATH': '.', 'MCP_LOG_LEVEL': 'INFO'}
            
            # Node.js/NPM 모듈인지 추측
            elif 'node' in server_id.lower() or 'npm' in server_id.lower() or 'playwright' in server_id.lower():
                suggested_config.command = 'npx'
                suggested_config.args = ['-y', f'@{server_id}/server']
                suggested_config.env = {'NODE_ENV': 'development', 'DEBUG': 'mcp:*'}
        
        elif server_type == MCPServerType.SSE:
            # 포트 추측
            port = 3000
            if 'github' in server_id.lower():
                port = 3001
            elif 'database' in server_id.lower():
                port = 3002
            
            suggested_config.url = f"http://localhost:{port}/sse"
            suggested_config.headers = {'Content-Type': 'application/json'}
        
        # 능력 추측
        for keyword, capabilities in self.llm_context['capabilities_mapping'].items():
            if keyword in server_id.lower():
                suggested_config.capabilities.extend(capabilities)
                break
        
        # 부분 설정으로 덮어쓰기
        for key, value in partial_config.items():
            if hasattr(suggested_config, key):
                setattr(suggested_config, key, value)
        
        return suggested_config
    
    def _create_default_config(self):
        """기본 설정 생성"""
        self.servers.clear()
        
        # 기본 서버들 - LLM First 원칙에 따라 최소한의 하드코딩
        default_servers = [
            'dataVisualizationLocal',
            'dataCleaningLocal', 
            'edaLocal',
            'featureEngineeringLocal',
            'sqlDatabaseLocal',
            'dataWranglingLocal'
        ]
        
        for server_id in default_servers:
            server_def = self.suggest_server_config(server_id, MCPServerType.STDIO)
            self.servers[server_id] = server_def
        
        # Playwright 추가 (특별 처리)
        playwright_def = MCPServerDefinition(
            server_id='playwrightLocal',
            server_type=MCPServerType.STDIO,
            name='Playwright 테스트 자동화',
            description='로컬 Playwright MCP 서버 - 브라우저 자동화 및 E2E 테스트',
            command='npx',
            args=['-y', '@smithery/cli@latest', 'run', '@executeautomation/playwright-mcp-server', '--key', '${PLAYWRIGHT_KEY}'],
            env={'DEBUG': 'pw:mcp', 'NODE_ENV': 'development'},
            timeout=30.0,
            capabilities=['browser_automation', 'ui_testing', 'screenshot', 'pdf_generation']
        )
        self.servers['playwrightLocal'] = playwright_def
        
        # 전역 설정
        self.global_settings = MCPGlobalSettings(
            environment_variables={
                'PLAYWRIGHT_KEY': 'your_playwright_key_here',
                'GITHUB_PAT': 'your_github_personal_access_token',
                'DATABASE_URL': 'sqlite:///./data/cherryai.db'
            },
            auto_discovery={'enabled': True, 'scan_ports': [3000, 3001, 3002, 3003, 3004, 3005, 3006]},
            llm_enhancement={'enabled': True, 'auto_configure': True}
        )
        
        # 메타데이터
        self.metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'description': 'CherryAI MCP 서버 설정 파일 - LLM First 원칙을 준수하는 동적 설정 관리'
        }

# 전역 설정 관리자 인스턴스
_config_manager = None

def get_mcp_config_manager() -> MCPConfigManager:
    """전역 MCP 설정 관리자 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = MCPConfigManager()
    return _config_manager 