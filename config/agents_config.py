"""
Dynamic Agent Configuration System
동적 에이전트 설정 관리 시스템

Features:
- JSON 기반 동적 에이전트 설정 로드
- 런타임 중 에이전트 추가/제거
- 설정 파일 변경 감지 및 자동 재로드
- 에이전트 상태 실시간 관리
"""

import json
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """에이전트 설정 데이터 모델"""
    id: str
    name: str
    display_name: str
    port: int
    host: str
    endpoint: str
    capabilities: List[str]
    description: str
    color: str
    category: str
    priority: int
    enabled: bool
    health_check_interval: int
    timeout: int
    retry_count: int

@dataclass
class AgentStatus:
    """에이전트 상태 정보"""
    agent_id: str
    name: str
    status: str  # 'online', 'offline', 'error', 'unknown'
    capabilities: List[str]
    current_load: float
    last_heartbeat: datetime
    performance_metrics: Dict[str, Any]

class AgentConfigLoader:
    """동적 에이전트 설정 로더"""
    
    def __init__(self, config_path: str = "config/agents.json"):
        self.config_path = Path(config_path)
        self.config_cache = {}
        self.last_modified = None
        self.global_settings = {}
        
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, AgentConfig]:
        """JSON 설정 파일에서 에이전트 설정 로드"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            return {}
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            agents = {}
            for agent_id, agent_data in config_data.get('agents', {}).items():
                try:
                    agents[agent_id] = AgentConfig(**agent_data)
                except TypeError as e:
                    logger.error(f"Invalid config for agent {agent_id}: {e}")
                    continue
            
            logger.info(f"Loaded {len(agents)} agent configurations")
            return agents
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def load_global_settings(self) -> Dict[str, Any]:
        """글로벌 설정 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data.get('global_settings', {})
        except Exception as e:
            logger.error(f"Error loading global settings: {e}")
            return {}
    
    async def watch_config_changes(self) -> None:
        """설정 파일 변경 감지 및 자동 재로드"""
        if not self.config_path.exists():
            return
            
        try:
            current_modified = self.config_path.stat().st_mtime
            if self.last_modified != current_modified:
                self.last_modified = current_modified
                await self.reload_config()
                logger.info("Config file changed, reloaded configuration")
        except Exception as e:
            logger.error(f"Error watching config changes: {e}")
    
    async def reload_config(self) -> Dict[str, AgentConfig]:
        """설정 재로드"""
        self.config_cache = self.load_config(str(self.config_path))
        self.global_settings = self.load_global_settings()
        return self.config_cache
    
    async def add_agent(self, agent_config: AgentConfig) -> bool:
        """새 에이전트 설정 추가"""
        try:
            with open(self.config_path, 'r+', encoding='utf-8') as f:
                config_data = json.load(f)
                config_data['agents'][agent_config.id] = asdict(agent_config)
                f.seek(0)
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                f.truncate()
                
            logger.info(f"Added agent {agent_config.id} to configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add agent {agent_config.id}: {e}")
            return False
    
    async def remove_agent(self, agent_id: str) -> bool:
        """에이전트 설정 제거"""
        try:
            with open(self.config_path, 'r+', encoding='utf-8') as f:
                config_data = json.load(f)
                if agent_id in config_data['agents']:
                    del config_data['agents'][agent_id]
                    f.seek(0)
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    f.truncate()
                    logger.info(f"Removed agent {agent_id} from configuration")
                    return True
                else:
                    logger.warning(f"Agent {agent_id} not found in configuration")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False
    
    async def update_agent_status(self, agent_id: str, enabled: bool) -> bool:
        """에이전트 활성화/비활성화 상태 업데이트"""
        try:
            with open(self.config_path, 'r+', encoding='utf-8') as f:
                config_data = json.load(f)
                if agent_id in config_data['agents']:
                    config_data['agents'][agent_id]['enabled'] = enabled
                    f.seek(0)
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    f.truncate()
                    logger.info(f"Updated agent {agent_id} status to {'enabled' if enabled else 'disabled'}")
                    return True
                else:
                    logger.warning(f"Agent {agent_id} not found in configuration")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id} status: {e}")
            return False
    
    def get_agents_by_category(self, category: str) -> List[AgentConfig]:
        """카테고리별 에이전트 목록 반환"""
        if not self.config_cache:
            self.config_cache = self.load_config(str(self.config_path))
        
        return [agent for agent in self.config_cache.values() 
                if agent.category == category and agent.enabled]
    
    def get_agents_by_capability(self, capability: str) -> List[AgentConfig]:
        """특정 능력을 가진 에이전트 목록 반환"""
        if not self.config_cache:
            self.config_cache = self.load_config(str(self.config_path))
        
        return [agent for agent in self.config_cache.values() 
                if capability in agent.capabilities and agent.enabled]
    
    def get_agent_by_id(self, agent_id: str) -> Optional[AgentConfig]:
        """ID로 에이전트 설정 반환"""
        if not self.config_cache:
            self.config_cache = self.load_config(str(self.config_path))
        
        return self.config_cache.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, AgentConfig]:
        """모든 에이전트 설정 반환"""
        if not self.config_cache:
            self.config_cache = self.load_config(str(self.config_path))
        
        return self.config_cache
    
    def get_enabled_agents(self) -> Dict[str, AgentConfig]:
        """활성화된 에이전트만 반환"""
        all_agents = self.get_all_agents()
        return {aid: agent for aid, agent in all_agents.items() if agent.enabled}

# 글로벌 설정 로더 인스턴스
agent_config_loader = AgentConfigLoader()

async def get_agent_config(agent_id: str) -> Optional[AgentConfig]:
    """에이전트 설정 조회 (비동기)"""
    return agent_config_loader.get_agent_by_id(agent_id)

async def get_all_enabled_agents() -> Dict[str, AgentConfig]:
    """활성화된 모든 에이전트 조회 (비동기)"""
    return agent_config_loader.get_enabled_agents()