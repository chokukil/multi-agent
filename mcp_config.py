"""
MCP Server Configuration Module
Centralized configuration for all MCP servers with environment variable support
"""

import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server"""
    name: str
    filename: str
    port: int
    description: str
    enabled: bool = True

class MCPConfigManager:
    """Manages MCP server configurations with environment variable support"""
    
    def __init__(self):
        self.servers = self._load_server_configs()
    
    def _get_port_from_env(self, server_key: str, default_port: int) -> int:
        """Get port from environment variable with fallback to default"""
        env_key = f"MCP_{server_key}_PORT"
        return int(os.getenv(env_key, default_port))
    
    def _load_server_configs(self) -> Dict[str, MCPServerConfig]:
        """Load all MCP server configurations based on actual files in mcp-servers folder"""
        
        # Define server configurations based on actual files in mcp-servers folder
        base_configs = [
            # File Management
            ("FILE_MANAGEMENT", "mcp-servers/mcp_file_management.py", 8006, "Safe file operations"),
            
            # Data Science and Analysis Tools
            ("DATA_SCIENCE", "mcp-servers/mcp_data_science_tools.py", 8007, "Comprehensive data science tools"),
            ("DATA_PREPROCESSING", "mcp-servers/mcp_data_preprocessing_tools.py", 8017, "Data preprocessing and cleaning tools"),
            ("STATISTICAL_ANALYSIS", "mcp-servers/mcp_statistical_analysis_tools.py", 8018, "Statistical analysis and hypothesis testing tools"),
            ("ADVANCED_ML_TOOLS", "mcp-servers/mcp_advanced_ml_tools.py", 8016, "Advanced machine learning tools"),
            
            # Semiconductor Analysis Servers
            ("SEMICONDUCTOR_YIELD", "mcp-servers/mcp_semiconductor_yield_analysis.py", 8008, "Semiconductor yield analysis"),
            ("PROCESS_CONTROL", "mcp-servers/mcp_process_control_charts.py", 8009, "Process control charts"),
            ("EQUIPMENT_ANALYSIS", "mcp-servers/mcp_semiconductor_equipment_analysis.py", 8010, "Equipment analysis"),
            ("DEFECT_PATTERN", "mcp-servers/mcp_defect_pattern_analysis.py", 8011, "Defect pattern analysis"),
            ("PROCESS_OPTIMIZATION", "mcp-servers/mcp_process_optimization.py", 8012, "Process optimization"),
            ("SEMICONDUCTOR_PROCESS", "mcp-servers/mcp_semiconductor_process_tools.py", 8020, "Comprehensive semiconductor process analysis tools"),
            
            # Time Series and Anomaly Detection
            ("TIMESERIES_ANALYSIS", "mcp-servers/mcp_timeseries_analysis.py", 8013, "Time series analysis specialist"),
            ("ANOMALY_DETECTION", "mcp-servers/mcp_anomaly_detection.py", 8014, "Anomaly detection specialist"),
            
            # Reporting Tools
            ("REPORT_WRITING", "mcp-servers/mcp_report_writing_tools.py", 8019, "Report writing and document generation tools"),
        ]
        
        servers = {}
        for key, filename, default_port, description in base_configs:
            port = self._get_port_from_env(key, default_port)
            
            # Check if server should be enabled (default: True)
            enabled_key = f"MCP_{key}_ENABLED"
            enabled = os.getenv(enabled_key, "true").lower() == "true"
            
            servers[key] = MCPServerConfig(
                name=key.replace('_', ' ').title(),
                filename=filename,
                port=port,
                description=description,
                enabled=enabled
            )
        
        return servers
    
    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get list of enabled servers"""
        return [server for server in self.servers.values() if server.enabled]
    
    def get_server_by_key(self, key: str) -> MCPServerConfig:
        """Get server configuration by key"""
        return self.servers.get(key)
    
    def get_all_servers(self) -> Dict[str, MCPServerConfig]:
        """Get all server configurations"""
        return self.servers
    
    def get_server_urls(self) -> List[str]:
        """Get list of enabled server URLs"""
        host = os.getenv("MCP_HOST", "localhost")
        enabled_servers = self.get_enabled_servers()
        return [f"http://{host}:{server.port}" for server in enabled_servers]
    
    def validate_server_files(self) -> Tuple[List[str], List[str]]:
        """Validate that server files exist"""
        existing = []
        missing = []
        
        for server in self.servers.values():
            if os.path.exists(server.filename):
                existing.append(server.filename)
            else:
                missing.append(server.filename)
        
        return existing, missing
    
    def print_configuration(self):
        """Print current MCP server configuration"""
        print("================================================")
        print("MCP Server Configuration")
        print("================================================")
        
        enabled_servers = self.get_enabled_servers()
        print(f"Enabled servers: {len(enabled_servers)}")
        print(f"Total servers: {len(self.servers)}")
        print()
        
        for key, server in self.servers.items():
            status = "✅ ENABLED" if server.enabled else "❌ DISABLED"
            file_status = "📁 EXISTS" if os.path.exists(server.filename) else "❌ MISSING"
            print(f"{server.name:<30} | Port {server.port:<4} | {status} | {file_status}")
            print(f"  └─ {server.description}")
            print(f"  └─ File: {server.filename}")
            print()

# Global instance
mcp_config = MCPConfigManager()

# Convenience functions
def get_mcp_servers() -> List[MCPServerConfig]:
    """Get list of enabled MCP servers"""
    return mcp_config.get_enabled_servers()

def get_mcp_server_urls() -> List[str]:
    """Get list of enabled MCP server URLs"""
    return mcp_config.get_server_urls()

def validate_mcp_files() -> Tuple[List[str], List[str]]:
    """Validate MCP server files exist"""
    return mcp_config.validate_server_files()

if __name__ == "__main__":
    # Print configuration when run directly
    mcp_config.print_configuration()
    
    existing, missing = validate_mcp_files()
    
    if missing:
        print("Missing files:")
        for file in missing:
            print(f"  ❌ {file}")
    
    if existing:
        print("Available files:")
        for file in existing:
            print(f"  ✅ {file}")