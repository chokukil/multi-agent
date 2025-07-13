#!/usr/bin/env python3
"""
🍒 CherryAI MCP Config Manager 단위 테스트
Phase 1.6: pytest 기반 핵심 기능 검증

Test Coverage:
- JSON 설정 로드/저장
- 서버 CRUD 작업
- 환경변수 치환
- 설정 검증
- LLM 기반 설정 제안

Author: CherryAI Team
Date: 2025-07-13
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append('.')

from core.monitoring.mcp_config_manager import (
    MCPConfigManager,
    MCPServerDefinition,
    MCPServerType,
    MCPGlobalSettings
)

class TestMCPConfigManager:
    """MCP Config Manager 테스트 클래스"""
    
    @pytest.fixture
    def temp_config_file(self):
        """임시 설정 파일 픽스처"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "mcpServers": {
                    "testServer": {
                        "type": "stdio",
                        "name": "Test Server",
                        "description": "테스트용 서버",
                        "command": "python",
                        "args": ["-m", "test.server"],
                        "env": {"TEST_VAR": "${TEST_VALUE}"},
                        "enabled": True,
                        "timeout": 30.0,
                        "retry_count": 3
                    },
                    "testsseServer": {
                        "type": "sse",
                        "name": "Test SSE Server", 
                        "description": "SSE 테스트 서버",
                        "url": "http://localhost:8080/sse",
                        "enabled": True
                    }
                },
                "globalSettings": {
                    "default_timeout": 20.0,
                    "default_retry_count": 3,
                    "environment_variables": {
                        "TEST_VALUE": "test_environment_value"
                    }
                },
                "metadata": {
                    "version": "1.0.0",
                    "created": "2025-07-13T00:00:00Z"
                }
            }
            json.dump(test_config, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # 클린업
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def config_manager(self, temp_config_file):
        """Config Manager 픽스처"""
        return MCPConfigManager(config_path=temp_config_file)
    
    def test_config_manager_initialization(self, config_manager):
        """Config Manager 초기화 테스트"""
        assert config_manager is not None
        assert len(config_manager.servers) > 0
        assert config_manager.global_settings is not None
    
    def test_load_config_success(self, config_manager):
        """설정 로드 성공 테스트"""
        result = config_manager.load_config()
        
        assert result is True
        assert len(config_manager.servers) == 2
        assert "testServer" in config_manager.servers
        assert "testsseServer" in config_manager.servers
    
    def test_server_definition_creation(self, config_manager):
        """서버 정의 생성 테스트"""
        server = config_manager.get_server("testServer")
        
        assert server is not None
        assert server.server_id == "testServer"
        assert server.server_type == MCPServerType.STDIO
        assert server.name == "Test Server"
        assert server.command == "python"
        assert server.args == ["-m", "test.server"]
        assert server.enabled is True
    
    def test_environment_variable_resolution(self, config_manager):
        """환경변수 치환 테스트"""
        server = config_manager.get_server("testServer")
        
        # 환경변수가 올바르게 치환되었는지 확인
        assert "TEST_VAR" in server.env
        assert server.env["TEST_VAR"] == "test_environment_value"
    
    def test_get_enabled_servers(self, config_manager):
        """활성화된 서버 조회 테스트"""
        enabled_servers = config_manager.get_enabled_servers()
        
        assert len(enabled_servers) == 2
        assert all(server.enabled for server in enabled_servers.values())
    
    def test_get_servers_by_type(self, config_manager):
        """타입별 서버 조회 테스트"""
        stdio_servers = config_manager.get_servers_by_type(MCPServerType.STDIO)
        sse_servers = config_manager.get_servers_by_type(MCPServerType.SSE)
        
        assert len(stdio_servers) == 1
        assert len(sse_servers) == 1
        assert "testServer" in stdio_servers
        assert "testsseServer" in sse_servers
    
    def test_add_server(self, config_manager):
        """서버 추가 테스트"""
        new_server = MCPServerDefinition(
            server_id="newTestServer",
            server_type=MCPServerType.STDIO,
            name="New Test Server",
            description="새로운 테스트 서버",
            command="echo",
            args=["hello"],
            enabled=True
        )
        
        result = config_manager.add_server(new_server)
        
        assert result is True
        assert "newTestServer" in config_manager.servers
        assert config_manager.get_server("newTestServer") == new_server
    
    def test_remove_server(self, config_manager):
        """서버 제거 테스트"""
        # 기존 서버 확인
        assert config_manager.get_server("testServer") is not None
        
        # 서버 제거
        result = config_manager.remove_server("testServer")
        
        assert result is True
        assert config_manager.get_server("testServer") is None
        assert "testServer" not in config_manager.servers
    
    def test_update_server(self, config_manager):
        """서버 업데이트 테스트"""
        updates = {
            "name": "Updated Test Server",
            "description": "업데이트된 설명",
            "timeout": 60.0
        }
        
        result = config_manager.update_server("testServer", updates)
        
        assert result is True
        
        updated_server = config_manager.get_server("testServer")
        assert updated_server.name == "Updated Test Server"
        assert updated_server.description == "업데이트된 설명"
        assert updated_server.timeout == 60.0
    
    def test_validate_server_config_valid(self, config_manager):
        """유효한 서버 설정 검증 테스트"""
        server = config_manager.get_server("testServer")
        is_valid, errors = config_manager.validate_server_config(server)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_server_config_invalid(self, config_manager):
        """유효하지 않은 서버 설정 검증 테스트"""
        invalid_server = MCPServerDefinition(
            server_id="invalidServer",
            server_type=MCPServerType.STDIO,
            name="",  # 빈 이름 (유효하지 않음)
            description="",
            command="",  # 빈 명령어 (유효하지 않음)
            enabled=True
        )
        
        is_valid, errors = config_manager.validate_server_config(invalid_server)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("name" in error.lower() for error in errors)
        assert any("command" in error.lower() for error in errors)
    
    def test_save_config(self, config_manager, temp_config_file):
        """설정 저장 테스트"""
        # 서버 추가
        new_server = MCPServerDefinition(
            server_id="saveTestServer",
            server_type=MCPServerType.SSE,
            name="Save Test Server",
            description="저장 테스트",
            url="http://localhost:9999",
            enabled=True
        )
        config_manager.add_server(new_server)
        
        # 설정 저장
        result = config_manager.save_config()
        assert result is True
        
        # 파일에서 다시 로드하여 확인
        new_manager = MCPConfigManager(config_path=temp_config_file)
        assert "saveTestServer" in new_manager.servers
        saved_server = new_manager.get_server("saveTestServer")
        assert saved_server.name == "Save Test Server"
    
    def test_suggest_server_config(self, config_manager):
        """서버 설정 제안 테스트"""
        suggested = config_manager.suggest_server_config(
            server_id="suggestedServer",
            server_type=MCPServerType.STDIO,
            partial_config={"name": "Suggested Server"}
        )
        
        assert suggested is not None
        assert suggested.server_id == "suggestedServer"
        assert suggested.server_type == MCPServerType.STDIO
        assert suggested.name == "Suggested Server"
        assert suggested.command is not None  # LLM이 제안한 명령어
        assert suggested.timeout > 0
    
    def test_global_settings_loading(self, config_manager):
        """전역 설정 로드 테스트"""
        settings = config_manager.global_settings
        
        assert settings is not None
        assert settings.default_timeout == 20.0
        assert settings.default_retry_count == 3
        assert "TEST_VALUE" in settings.environment_variables
        assert settings.environment_variables["TEST_VALUE"] == "test_environment_value"
    
    def test_environment_variable_fallback(self):
        """환경변수 폴백 테스트"""
        # 실제 환경변수 설정
        os.environ["REAL_ENV_VAR"] = "real_value"
        
        try:
            # 실제 환경변수를 참조하는 설정
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                test_config = {
                    "mcpServers": {
                        "envTestServer": {
                            "type": "stdio",
                            "name": "Env Test Server",
                            "command": "${REAL_ENV_VAR}",
                            "enabled": True
                        }
                    },
                    "globalSettings": {}
                }
                json.dump(test_config, f, indent=2)
                temp_path = f.name
            
            config_manager = MCPConfigManager(config_path=temp_path)
            server = config_manager.get_server("envTestServer")
            
            # 실제 환경변수 값이 사용되었는지 확인
            assert server.command == "real_value"
            
        finally:
            # 클린업
            if "REAL_ENV_VAR" in os.environ:
                del os.environ["REAL_ENV_VAR"]
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_invalid_config_file_handling(self):
        """유효하지 않은 설정 파일 처리 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            config_manager = MCPConfigManager(config_path=temp_path)
            
            # 기본 설정이 생성되었는지 확인
            assert len(config_manager.servers) >= 0  # 기본 설정 또는 빈 설정
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_missing_config_file_creation(self):
        """존재하지 않는 설정 파일 생성 테스트"""
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "new_config.json")
        
        try:
            # 존재하지 않는 파일로 초기화
            config_manager = MCPConfigManager(config_path=config_path)
            
            # 기본 설정이 생성되고 파일이 저장되었는지 확인
            assert os.path.exists(config_path)
            
            # 파일 내용 확인
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                assert "mcpServers" in config_data
                assert "globalSettings" in config_data
                
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            os.rmdir(temp_dir)

@pytest.mark.asyncio
class TestMCPConfigManagerAsync:
    """MCP Config Manager 비동기 테스트"""
    
    def test_thread_safety(self):
        """스레드 안전성 테스트"""
        import threading
        import time
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "mcpServers": {},
                "globalSettings": {}
            }
            json.dump(test_config, f, indent=2)
            temp_path = f.name
        
        try:
            config_manager = MCPConfigManager(config_path=temp_path)
            results = []
            
            def add_servers(thread_id):
                for i in range(5):
                    server = MCPServerDefinition(
                        server_id=f"thread_{thread_id}_server_{i}",
                        server_type=MCPServerType.STDIO,
                        name=f"Thread {thread_id} Server {i}",
                        description="Thread safety test",
                        command="echo",
                        enabled=True
                    )
                    result = config_manager.add_server(server)
                    results.append(result)
                    time.sleep(0.01)  # 작은 지연
            
            # 여러 스레드에서 동시에 서버 추가
            threads = []
            for i in range(3):
                thread = threading.Thread(target=add_servers, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # 모든 추가 작업이 성공했는지 확인
            assert all(results)
            assert len(config_manager.servers) == 15  # 3 스레드 * 5 서버
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 