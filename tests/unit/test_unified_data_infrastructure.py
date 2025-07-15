"""
통합 데이터 시스템 인프라 단위 테스트

pandas_agent 패턴을 기준으로 한 통합 인프라 컴포넌트들의 동작을 검증
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 테스트 대상 import
from a2a_ds_servers.unified_data_system.core.unified_data_interface import (
    UnifiedDataInterface, DataIntent, DataIntentType, DataProfile, QualityReport, A2AContext
)
from a2a_ds_servers.unified_data_system.core.llm_first_data_engine import LLMFirstDataEngine
from a2a_ds_servers.unified_data_system.utils.file_scanner import FileScanner
from a2a_ds_servers.unified_data_system.utils.encoding_detector import EncodingDetector


class TestDataTypes:
    """데이터 타입들의 기본 동작 테스트"""
    
    def test_data_intent_creation(self):
        """DataIntent 객체 생성 테스트"""
        intent = DataIntent(
            intent_type=DataIntentType.ANALYSIS,
            confidence=0.9,
            file_preferences=["test.csv"],
            operations=["analyze", "visualize"],
            constraints={"max_rows": 1000}
        )
        
        assert intent.intent_type == DataIntentType.ANALYSIS
        assert intent.confidence == 0.9
        assert intent.file_preferences == ["test.csv"]
        assert intent.operations == ["analyze", "visualize"]
        assert intent.constraints == {"max_rows": 1000}
        assert intent.priority == 1  # 기본값
        assert intent.requires_visualization == False  # 기본값
    
    def test_data_intent_types(self):
        """모든 DataIntentType 열거형 테스트"""
        expected_types = {
            "analysis", "visualization", "cleaning", "transformation",
            "modeling", "feature_engineering", "sql_query", 
            "reporting", "eda", "orchestration"
        }
        
        actual_types = {intent.value for intent in DataIntentType}
        assert actual_types == expected_types
    
    def test_a2a_context_creation(self):
        """A2AContext 생성 및 정보 추출 테스트"""
        mock_context = Mock()
        mock_context.request = {
            "session_id": "test_session_123",
            "user_id": "user_456",
            "request_id": "req_789"
        }
        
        a2a_context = A2AContext(mock_context)
        
        assert a2a_context.session_id == "test_session_123"
        assert a2a_context.user_id == "user_456"
        assert a2a_context.request_id == "req_789"
        
        context_dict = a2a_context.to_dict()
        assert "timestamp" in context_dict
        assert context_dict["session_id"] == "test_session_123"


class TestFileScanner:
    """FileScanner 단위 테스트"""
    
    @pytest.fixture
    def file_scanner(self):
        return FileScanner()
    
    @pytest.fixture
    def temp_data_files(self):
        """테스트용 임시 데이터 파일들 생성"""
        temp_dir = tempfile.mkdtemp()
        
        # 다양한 형식의 테스트 파일 생성
        test_files = {
            "test.csv": "name,age,city\nJohn,25,Seoul\nJane,30,Busan",
            "test.xlsx": "dummy_excel_content",
            "test.json": '{"data": [{"id": 1, "value": "test"}]}',
            "test.txt": "This is a test file",
            "invalid.xyz": "unsupported format"
        }
        
        created_files = []
        for filename, content in test_files.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(file_path)
        
        yield temp_dir, created_files
        
        # 정리
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_scan_directory(self, file_scanner, temp_data_files):
        """디렉토리 스캔 기능 테스트"""
        temp_dir, created_files = temp_data_files
        
        # 실제 스캔 실행
        scanned_files = await file_scanner._scan_directory(temp_dir)
        
        # 지원되는 파일만 스캔되었는지 확인
        scanned_names = [Path(f).name for f in scanned_files]
        
        assert "test.csv" in scanned_names
        assert "test.json" in scanned_names
        assert "test.txt" in scanned_names
        assert "invalid.xyz" not in scanned_names  # 지원되지 않는 형식
    
    @pytest.mark.asyncio
    async def test_file_info_retrieval(self, file_scanner, temp_data_files):
        """파일 정보 조회 테스트"""
        temp_dir, created_files = temp_data_files
        csv_file = next(f for f in created_files if f.endswith('.csv'))
        
        file_info = await file_scanner.get_file_info(csv_file)
        
        assert file_info["name"] == "test.csv"
        assert file_info["extension"] == ".csv"
        assert file_info["size_bytes"] > 0
        assert file_info["is_readable"] == True
        assert file_info["file_type"] == "CSV file"
    
    def test_supported_extensions(self, file_scanner):
        """지원되는 파일 확장자 테스트"""
        extensions = file_scanner.get_supported_extensions()
        
        expected_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet', '.feather', '.txt', '.tsv'}
        actual_extensions = set(extensions.keys())
        
        assert expected_extensions.issubset(actual_extensions)
    
    @pytest.mark.asyncio
    async def test_file_validation(self, file_scanner, temp_data_files):
        """파일 접근 가능성 검증 테스트"""
        temp_dir, created_files = temp_data_files
        csv_file = next(f for f in created_files if f.endswith('.csv'))
        
        validation = await file_scanner.validate_file_access(csv_file)
        
        assert validation["exists"] == True
        assert validation["readable"] == True
        assert validation["is_supported"] == True
        assert validation["size_ok"] == True
        assert len(validation["errors"]) == 0


class TestEncodingDetector:
    """EncodingDetector 단위 테스트"""
    
    @pytest.fixture
    def encoding_detector(self):
        return EncodingDetector()
    
    @pytest.fixture
    def temp_encoded_files(self):
        """다양한 인코딩의 테스트 파일들 생성"""
        temp_dir = tempfile.mkdtemp()
        
        # 테스트 텍스트 (한글 포함)
        test_text = "안녕하세요, Hello World! 테스트 데이터입니다."
        
        # 다양한 인코딩으로 파일 생성
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        created_files = {}
        
        for encoding in encodings:
            try:
                file_path = os.path.join(temp_dir, f"test_{encoding}.txt")
                
                if encoding == 'latin1':
                    # latin1은 한글을 지원하지 않으므로 영문만
                    content = "Hello World! Test data."
                else:
                    content = test_text
                
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                created_files[encoding] = file_path
            except UnicodeEncodeError:
                # 인코딩이 텍스트를 지원하지 않는 경우 스킵
                pass
        
        yield temp_dir, created_files
        
        # 정리
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_encoding_detection(self, encoding_detector, temp_encoded_files):
        """인코딩 자동 감지 테스트"""
        temp_dir, created_files = temp_encoded_files
        
        for expected_encoding, file_path in created_files.items():
            detected_encoding = await encoding_detector.detect_encoding(file_path)
            
            # 정확한 감지 또는 호환 가능한 인코딩인지 확인
            assert detected_encoding is not None
            assert isinstance(detected_encoding, str)
            assert len(detected_encoding) > 0
    
    @pytest.mark.asyncio
    async def test_encoding_testing(self, encoding_detector, temp_encoded_files):
        """특정 인코딩 테스트 기능"""
        temp_dir, created_files = temp_encoded_files
        
        # UTF-8 파일에 대해 다양한 인코딩 테스트
        if 'utf-8' in created_files:
            utf8_file = created_files['utf-8']
            
            # UTF-8로 읽기 성공해야 함
            result = await encoding_detector._test_encoding(utf8_file, 'utf-8')
            assert result == True
            
            # 잘못된 인코딩으로는 실패할 가능성 높음 (하지만 호환될 수도 있음)
            # 따라서 결과가 boolean인지만 확인
            result = await encoding_detector._test_encoding(utf8_file, 'cp949')
            assert isinstance(result, bool)
    
    def test_encoding_priority_list(self, encoding_detector):
        """인코딩 우선순위 리스트 테스트"""
        priority = encoding_detector.get_encoding_priority()
        
        # 기본 인코딩들이 포함되어 있는지 확인
        expected_encodings = ['utf-8', 'cp949', 'euc-kr']
        for encoding in expected_encodings:
            assert encoding in priority
        
        # UTF-8이 첫 번째 우선순위인지 확인
        assert priority[0] == 'utf-8'
    
    @pytest.mark.asyncio
    async def test_encoding_candidates(self, encoding_detector, temp_encoded_files):
        """인코딩 후보 리스트 생성 테스트"""
        temp_dir, created_files = temp_encoded_files
        
        if created_files:
            file_path = next(iter(created_files.values()))
            candidates = await encoding_detector.get_encoding_candidates(file_path)
            
            assert isinstance(candidates, list)
            assert len(candidates) > 0
            
            # 각 후보가 필요한 정보를 포함하는지 확인
            for candidate in candidates:
                assert "encoding" in candidate
                assert "confidence" in candidate
                assert "tested" in candidate
                assert isinstance(candidate["tested"], bool)


class TestLLMFirstDataEngine:
    """LLMFirstDataEngine 단위 테스트"""
    
    @pytest.fixture
    def llm_engine(self):
        # Mock LLM을 사용하여 실제 API 호출 없이 테스트
        return LLMFirstDataEngine()
    
    @pytest.fixture
    def mock_context(self):
        """테스트용 Mock A2A Context"""
        mock_context = Mock()
        mock_context.request = {
            "session_id": "test_session",
            "user_id": "test_user"
        }
        mock_context.message = Mock()
        mock_context.message.parts = []
        
        return A2AContext(mock_context)
    
    def test_default_model_selection(self, llm_engine):
        """기본 LLM 모델 선택 로직 테스트"""
        # 환경변수가 없는 상태에서 폴백 모델 선택 확인
        assert llm_engine.model_name is not None
        assert isinstance(llm_engine.model_name, str)
        assert len(llm_engine.model_name) > 0
    
    @pytest.mark.asyncio
    async def test_fallback_intent_analysis(self, llm_engine, mock_context):
        """폴백 의도 분석 테스트 (LLM 없이)"""
        test_queries = {
            "데이터를 시각화해주세요": DataIntentType.VISUALIZATION,
            "데이터를 정리하고 싶어요": DataIntentType.CLEANING,
            "모델을 만들어주세요": DataIntentType.MODELING,
            "탐색적 분석을 해주세요": DataIntentType.EDA,
            "일반적인 분석": DataIntentType.ANALYSIS
        }
        
        for query, expected_type in test_queries.items():
            intent = llm_engine._fallback_intent_analysis(query)
            
            assert isinstance(intent, DataIntent)
            assert intent.intent_type == expected_type
            assert intent.confidence > 0
            assert isinstance(intent.operations, list)
    
    def test_fallback_file_selection(self, llm_engine):
        """폴백 파일 선택 로직 테스트"""
        test_files = [
            "/path/to/data.json",
            "/path/to/data.csv", 
            "/path/to/data.xlsx",
            "/path/to/data.parquet"
        ]
        
        # 분석 의도에 대한 파일 선택
        intent = DataIntent(
            intent_type=DataIntentType.ANALYSIS,
            confidence=0.8,
            file_preferences=[],
            operations=["analyze"],
            constraints={}
        )
        
        selected_file = llm_engine._fallback_file_selection(intent, test_files)
        
        assert selected_file in test_files
        # CSV가 분석에 우선되는지 확인
        assert selected_file.endswith('.csv')
    
    def test_fallback_loading_strategy(self, llm_engine):
        """폴백 로딩 전략 생성 테스트"""
        # 다양한 파일 크기에 대한 전략 테스트
        test_cases = [
            {"size_mb": 10, "expected_chunk": None},      # 소형 파일
            {"size_mb": 200, "expected_chunk": 5000},     # 중형 파일  
            {"size_mb": 800, "expected_chunk": 10000}     # 대형 파일
        ]
        
        for case in test_cases:
            file_info = {"size_mb": case["size_mb"]}
            intent = DataIntent(
                intent_type=DataIntentType.ANALYSIS,
                confidence=0.8,
                file_preferences=[],
                operations=["analyze"],
                constraints={}
            )
            
            strategy = llm_engine._create_fallback_strategy(file_info, intent)
            
            assert strategy.encoding == 'utf-8'
            assert strategy.use_cache == True
            assert strategy.chunk_size == case["expected_chunk"]


@pytest.mark.asyncio
async def test_unified_data_interface_abstract():
    """UnifiedDataInterface 추상 클래스 테스트"""
    
    # 추상 클래스 직접 인스턴스화는 불가능해야 함
    with pytest.raises(TypeError):
        UnifiedDataInterface()


class MockUnifiedAgent(UnifiedDataInterface):
    """테스트용 UnifiedDataInterface 구현체"""
    
    async def load_data(self, intent, context):
        return Mock()  # SmartDataFrame 모킹
    
    async def get_data_info(self):
        return DataProfile(
            shape=(100, 5),
            dtypes={"col1": "int64", "col2": "object"},
            missing_values={"col1": 0, "col2": 2},
            memory_usage=1024,
            encoding="utf-8",
            file_size=2048
        )
    
    async def validate_data_quality(self):
        return QualityReport(
            overall_score=0.95,
            completeness=0.98,
            consistency=0.92,
            validity=0.96,
            accuracy=0.94,
            uniqueness=0.90,
            issues=[],
            recommendations=["데이터 품질이 양호합니다"],
            passed_checks=["no_missing_values", "valid_types"],
            failed_checks=[]
        )


@pytest.mark.asyncio
async def test_mock_unified_agent():
    """Mock UnifiedDataInterface 구현체 테스트"""
    agent = MockUnifiedAgent()
    
    # 기본 메서드들이 올바르게 작동하는지 확인
    capabilities = agent.get_agent_capabilities()
    assert capabilities["llm_first"] == True
    assert capabilities["a2a_compliant"] == True
    
    # 데이터 프로파일 테스트
    profile = await agent.get_data_info()
    assert profile.shape == (100, 5)
    assert profile.encoding == "utf-8"
    
    # 품질 리포트 테스트  
    quality = await agent.validate_data_quality()
    assert quality.overall_score == 0.95
    assert len(quality.passed_checks) > 0


def test_infrastructure_integration():
    """인프라 컴포넌트들의 통합 테스트"""
    # 모든 핵심 컴포넌트가 import 가능한지 확인
    from a2a_ds_servers.unified_data_system.core.unified_data_interface import UnifiedDataInterface
    from a2a_ds_servers.unified_data_system.core.llm_first_data_engine import LLMFirstDataEngine
    from a2a_ds_servers.unified_data_system.utils.file_scanner import FileScanner
    from a2a_ds_servers.unified_data_system.utils.encoding_detector import EncodingDetector
    
    # 각 컴포넌트가 올바르게 인스턴스화되는지 확인
    llm_engine = LLMFirstDataEngine()
    file_scanner = FileScanner()
    encoding_detector = EncodingDetector()
    
    assert llm_engine is not None
    assert file_scanner is not None
    assert encoding_detector is not None
    
    # 기본 설정들이 올바른지 확인
    assert len(file_scanner.get_supported_extensions()) > 0
    assert len(encoding_detector.get_encoding_priority()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 