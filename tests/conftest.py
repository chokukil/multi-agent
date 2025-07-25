"""
Pytest configuration and fixtures for Cherry AI Platform tests.
"""

import pytest
import sys
import os
import tempfile
import shutil
import asyncio
import socket
from contextlib import closing
from unittest.mock import Mock, MagicMock
from datetime import datetime
import pandas as pd

import pytest_asyncio
from uvicorn import Config, Server

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules for testing
try:
    from modules.models import VisualDataCard, EnhancedChatMessage, EnhancedArtifact, DataQualityInfo
    from modules.core.security_validation_system import LLMSecurityValidationSystem, SecurityContext
    from modules.ui.user_experience_optimizer import UserExperienceOptimizer, UserProfile
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

REGISTRY_HOST = "127.0.0.1"


class TestServer(Server):
    def __init__(self, app, host, port):
        self._startup_done = asyncio.Event()
        super().__init__(config=Config(app, host=host, port=port))

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        self._startup_done.set()

    async def _serve(self):
        try:
            await self.serve()
        except asyncio.CancelledError:
            await self.shutdown()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['Engineering', 'Marketing', 'Sales', 'HR', 'Engineering']
    })


@pytest.fixture
def sample_data_card(sample_dataframe):
    """Create a sample VisualDataCard for testing."""
    return VisualDataCard(
        id="test_card_1",
        name="Test Dataset",
        file_path="/test/path/data.csv",
        format="CSV",
        rows=len(sample_dataframe),
        columns=len(sample_dataframe.columns),
        memory_usage="2.5 KB",
        preview=sample_dataframe.head(),
        metadata={
            'upload_time': datetime.now().isoformat(),
            'column_names': sample_dataframe.columns.tolist(),
            'column_types': sample_dataframe.dtypes.to_dict()
        },
        quality_indicators=DataQualityInfo(
            quality_score=85.0,
            completeness=0.95,
            consistency=0.90,
            validity=0.88,
            issues=[]
        )
    )


@pytest.fixture
def mock_security_context():
    """Create a mock security context."""
    return SecurityContext(
        user_id="test_user",
        session_id="test_session",
        ip_address="127.0.0.1",
        user_agent="Test Agent",
        timestamp=datetime.now(),
        request_count=1,
        risk_score=0.0,
        previous_violations=[]
    )


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
    mock_st = Mock()
    mock_st.session_state = {}
    mock_st.empty = Mock(return_value=Mock())
    mock_st.spinner = Mock()
    mock_st.success = Mock()
    mock_st.error = Mock()
    mock_st.warning = Mock()
    mock_st.info = Mock()
    mock_st.markdown = Mock()
    mock_st.dataframe = Mock()
    mock_st.json = Mock()
    return mock_st


@pytest.fixture(autouse=True)
def mock_universal_engine():
    """Mock Universal Engine components for testing."""
    import sys
    from unittest.mock import MagicMock
    
    # Create mock modules
    mock_modules = [
        'core.universal_engine.meta_reasoning_engine',
        'core.universal_engine.llm_factory',
        'core.universal_engine.a2a_integration.a2a_workflow_orchestrator',
        'core.universal_engine.intelligent_error_diagnosis',
        'core.universal_engine.context_aware_recovery',
        'core.universal_engine.self_healing_system',
    ]
    
    for module_name in mock_modules:
        sys.modules[module_name] = MagicMock()


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'test_data_dir': 'tests/test_data',
        'max_test_file_size': 10 * 1024 * 1024,  # 10MB
        'test_timeout': 30,  # seconds
        'mock_llm_responses': True,
        'skip_slow_tests': False
    }


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    markers = [
        "unit: Unit tests",
        "integration: Integration tests", 
        "security: Security tests",
        "performance: Performance tests",
        "e2e: End-to-end tests",
        "slow: Slow running tests",
        "requires_llm: Tests that require LLM API",
        "requires_network: Tests that require network access"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)
 