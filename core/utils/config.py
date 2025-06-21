# File: utils/config.py
# Location: ./utils/config.py

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

class Config:
    """Configuration management for the multi-agent system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv("CONFIG_FILE", "config.json")
        self.config = self._load_config()
    
    def reload(self):
        """Reloads the configuration from file and environment."""
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        config = {
            # LLM Settings
            "llm": {
                "provider": os.getenv("LLM_PROVIDER", "OPENAI"),
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
                },
                "ollama": {
                    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    "model": os.getenv("OLLAMA_MODEL", "llama2")
                }
            },
            
            # System Settings
            "system": {
                "recursion_limit": int(os.getenv("RECURSION_LIMIT", "30")),
                "timeout_seconds": int(os.getenv("TIMEOUT_SECONDS", "180")),
                "max_retries": int(os.getenv("MAX_RETRIES", "3")),
                "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true"
            },
            
            # Data Settings
            "data": {
                "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "100")),
                "allowed_extensions": ["csv", "xlsx", "xls", "json"],
                "datasets_dir": "./sandbox/datasets"
            },
            
            # Logging Settings
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "file": os.getenv("LOG_FILE", "debug.log"),
                "max_logs": int(os.getenv("MAX_LOGS", "1000"))
            },
            
            # Langfuse Settings
            "langfuse": {
                "host": os.getenv("LANGFUSE_HOST"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY")
            },
            
            # UI Settings
            "ui": {
                "theme": os.getenv("UI_THEME", "light"),
                "max_chat_history": int(os.getenv("MAX_CHAT_HISTORY", "50")),
                "show_advanced": os.getenv("SHOW_ADVANCED", "false").lower() == "true"
            }
        }
        
        # Try to load from file
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with environment config (env takes precedence)
                    config = self._deep_merge(file_config, config)
            except Exception as e:
                print(f"Failed to load config file: {e}")
        
        return config
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key"""
        # if '.' not in key, return the whole dictionary
        if '.' not in key:
            return self.config.get(key, default)

        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-separated key"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, file_path: Optional[str] = None):
        """Save configuration to file"""
        file_path = file_path or self.config_file
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # Check required fields
        if not self.get('llm.openai.api_key') and self.get('llm.provider') == 'OPENAI':
            errors.append("OpenAI API key is required when using OpenAI provider")
        
        # Check value ranges
        if not 5 <= self.get('system.recursion_limit', 30) <= 100:
            errors.append("Recursion limit should be between 5 and 100")
        
        if not 30 <= self.get('system.timeout_seconds', 180) <= 3600:
            errors.append("Timeout should be between 30 and 3600 seconds")
        
        return len(errors) == 0, errors

# Global config instance
config = Config()

# MCP Configuration Management Functions
def load_mcp_configs() -> List[Dict[str, Any]]:
    """Load all MCP configurations from mcp-configs directory"""
    mcp_config_dir = Path("mcp-configs")
    configs = []
    
    if not mcp_config_dir.exists():
        mcp_config_dir.mkdir(exist_ok=True)
        return configs
    
    for json_file in mcp_config_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                config_data['config_name'] = json_file.stem
                configs.append(config_data)
        except Exception as e:
            print(f"Failed to load MCP config {json_file}: {e}")
    
    return configs

def save_mcp_config(config_name: str, config_data: Dict[str, Any]) -> bool:
    """Save MCP configuration to file"""
    try:
        mcp_config_dir = Path("mcp-configs")
        mcp_config_dir.mkdir(exist_ok=True)
        
        config_file = mcp_config_dir / f"{config_name}.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Failed to save MCP config {config_name}: {e}")
        return False

def delete_mcp_config(config_name: str) -> bool:
    """Delete MCP configuration file"""
    try:
        mcp_config_dir = Path("mcp-configs")
        config_file = mcp_config_dir / f"{config_name}.json"
        
        if config_file.exists():
            config_file.unlink()
            return True
        return False
    except Exception as e:
        print(f"Failed to delete MCP config {config_name}: {e}")
        return False

def get_mcp_config(config_name: str) -> Optional[Dict[str, Any]]:
    """Get specific MCP configuration"""
    try:
        mcp_config_dir = Path("mcp-configs")
        config_file = mcp_config_dir / f"{config_name}.json"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Failed to get MCP config {config_name}: {e}")
        return None

# Helper functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key, default)

def reload_config():
    """Helper function to reload the global configuration object."""
    config.reload()

def set_config(key: str, value: Any):
    """Set configuration value"""
    config.set(key, value)

def validate_config() -> Tuple[bool, List[str]]:
    """Validate configuration"""
    return config.validate()